# aiter on gfx1201 (Radeon AI PRO R9700): Phase 2 report

This covers the Phase 2 order in [`gfx1201_status.md`](gfx1201_status.md) section 5. It was
measured on one R9700 (gfx1201, 32 GB, wave32) in a vLLM ROCm image with torch 2.13 /
HIP 7.15 / Triton 3.8.0, built for `GPU_ARCHS=gfx1201` only.

Every change is a separate commit on `gfx1201-phase2`, and each commit message carries its
own before/after evidence. Nothing was relaxed to make a test pass: no tolerance loosened, no
case skipped, no test disabled.

## 1. What landed, and what did not

| # | Step | Result | Commit |
|---|---|---|---|
| 0 | Working timer | `run_perftest` falls back to HIP events when the profiler records no device kernels | `e8a2808fe` |
| 1 | Rope triage | **Unresolved** (section 3.1) | none |
| 2 | Tier A seeds | `gemm_a16w16`, `_persistent`, `_atomic`, `_gated`, `fused_gemm_a16w16_mul_add`, `gemm_a16w4` landed; `batched_gemm_a16w16` and `ff_a16w16_fused` **not merged** (section 2) | `bca75b2a5`, `63d91823a`, `9fca7c3b4`, `02033177c`, `f961a0326`, `64e30009c` |
| 3 | Tuner wave size | `exceeds_registers` counts 32 lanes on RDNA | `e4e49010f` |
| 4 | topk LDS | one-block radix kernel fits 64 KiB (50,432 B peak) | `e0706983f` |
| 5 | FlyDSL GDR wave32 | correct but **not faster**: kept on an unpublished local branch, not merged (section 3.5) | none |
| 6 | One-line gates | qk_norm_rope_quant w32 builder, opus packed atomics, rope OCP FP8 pack: three commits rather than one, so each carries its own evidence (squash if preferred) | `42fa0af7b`, `118d8947a`, `b29d8739f` |
| 7 | OCP-FP8 MoE case | `test_moe_gemm_a8w8.py` runs float8_e4m3fn on gfx1200/gfx1201: 64 pass, 0 before | `824622996` |
| 8 | FlyDSL `causal_conv1d_update` | arch admitted; 40/41 pass, faster than every baseline in 366 of 420 cells; one bit-exactness assertion fails (section 3.6) | `bee66aaec` |
| 9 | Sweeps | only the Tier A seeds above. No blockscale sweep: no FP8-blockscale checkpoint is in use | none |

## 2. Seeds

Each seed comes from a sweep per M bucket. The candidates are the arch seeds the family
already has plus the top ten `gemm_a16w16` sweep configs that fit 64 KiB of LDS and the
register budget. Each candidate is checked against an fp32 reference and scored by geomean
speedup over torch on Qwen3.8-27B projection shapes (`do_bench` median, L2 flushed).
`gemm_a16w4` is compared with `F.linear` on weights already dequantized to bf16, which is the
fastest torch can do and therefore conservative.

| family | M <= 128 | M > 128 | merged |
|---|---|---|---|
| `gemm_a16w16` | x1.08-x1.33 (gdn_ba N=96 loses at M >= 24) | x0.69-x1.03 | yes |
| `gemm_a16w16_persistent` | x1.04-x1.33 | x0.60-x0.71 | yes |
| `gemm_a16w16_atomic` | x1.16-x1.39 | x0.72-x0.91 | yes |
| `gemm_a16w16_gated` | x1.21-x1.29 | x0.94-x1.26 | yes |
| `fused_gemm_a16w16_mul_add` | x1.18-x1.45 | x0.83-x0.99 | yes |
| `gemm_a16w4` | x1.48-x2.35 | x0.80-x1.07 | yes |
| `batched_gemm_a16w16` | x0.89-x1.19 | x0.90-x1.21 | **no**: a toss-up against `torch.bmm`, and no Qwen consumer |
| `ff_a16w16_fused` | x1.05 at M <= 4, x0.13-x0.82 above | x0.07-x0.11 | **no**: slower than torch |

**The merged seeds are slower than torch at large M.** Without a gfx1201 config every one of
these ops asserted, so a seed is strictly better than nothing for a caller that has already
chosen aiter. But a caller choosing between aiter and torch at M > 128 should choose torch. The
review question is whether to keep the large-M buckets or let those shapes fall through. The
loader has no fall-through today: a missing bucket uses `any`.

Op tests with the seeds (Phase 1 baseline in brackets):

| test file | pass | fail | skip |
|---|---:|---:|---:|
| `triton_tests/gemm/basic/test_gemm_a16w16.py` | 377 [5] | 0 [372] | 764 [764] |
| `triton_tests/gemm/basic/test_gemm_a16w16_gated.py` | 900 [0] | 0 [900] | 1,692 [1,692] |
| `triton_tests/gemm/fused/test_fused_gemm_a16w16_mul_add.py` | 216 [0] | 0 [216] | 0 [0] |
| `triton_tests/gemm/basic/test_gemm_a16w4.py` | 356 [0] | 0 [0] | 8 [364] |

`test_gemm_a16w4.py` skipped every case off gfx942/gfx1100 because no config existed. Its
`SUPPORTED_ARCHS` gains gfx1201 in the seed's commit. The 8 skips left are the test's own
gptq_v2 rule.

## 3. Findings

### 3.1 Rope mismatches: unresolved

The 2,091 Phase 1 rope mismatches (about 1 bf16 ULP) do not come from the store. Triton's bf16
store on gfx1201 rounds to nearest even and is 100 % equal to torch's rounding of the same fp32
values. No candidate model of the kernel's arithmetic matches it on more than 68 % of elements;
per-op bf16 truncation is the closest. The deviation is in the kernel's arithmetic, and pinning
it down needs a read of the generated ISA. No tolerance was changed.

### 3.2 Opus

- `_atomic_pk_add_cas`, the compare-and-swap fallback for packed f16/bf16 atomic add, does not
  compile for a 2 x 16-bit vector. It bit-casts a 2-byte `_load<1>` to a 32-bit integer. The
  gfx1201 commit avoids it by enabling the native instruction, but it is still reached on archs
  without that instruction (bf16 on gfx942, for one).
- `OPUS_HAS_BUFFER_ATOMIC_FMINMAX_F32` is defined in `opus.hpp` and read nowhere.

### 3.3 `gemm_a16w16_atomic` split-K double count (all archs)

The kernel's K loop runs `for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter)` with
pointers that already start at `pid_k * SPLITK_BLOCK_SIZE`. When `cdiv(K, NUM_KSPLIT)` is not a
multiple of `BLOCK_SIZE_K`, splits overlap or run past their range, and the result is wrong by
order 1 (errors of 2.9-3.0 against 0.015 for torch). The sweep's accuracy check rejected every
such config, and every merged atomic bucket uses `NUM_KSPLIT=1`. A user-supplied config with
`NUM_KSPLIT > 1` can still hit it.

### 3.4 Sweep accuracy gate for `ff_a16w16_fused`

The fused feed-forward kernel accumulates its output with bf16 atomic adds across intermediate
tiles, by design. A gate of "within 2x torch's own error" rejects every config, including the
existing arch seeds. That sweep therefore uses the repo test's own criterion (`ff_gated_test`:
`assert_close` against torch's bf16 output, atol = rtol = 0.1). It still loses to torch (section 2).

### 3.5 FlyDSL GDR decode on wave32: not merged

The port is correct, with error 0 against the reference on every cell. Median us, num_v_heads 48, bf16:

| b | s | FlyDSL direct | public entry, FlyDSL | public entry, Triton |
|---:|---:|---:|---:|---:|
| 1 | 1 | 37.4 | 65.8 | 121.2 |
| 3 | 1 | 94.7 | 160.0 | 128.8 |
| 6 | 1 | 44.4 | 75.7 | 59.1 |
| 32 | 1 | 374.4 | 376.7 | 360.2 |
| 1 | 2 | 33.1 | 73.1 | 62.9 |
| 3 | 2 | 39.4 | 78.8 | 59.8 |
| 6 | 2 | 46.8 | 70.6 | 64.9 |
| 32 | 2 | 570.8 | 571.9 | 560.1 |
| 1 | 8 | 34.5 | 72.1 | 57.3 |
| 3 | 8 | 95.8 | 107.6 | 104.1 |
| 6 | 8 | 376.6 | 366.4 | 313.2 |
| 32 | 8 | 1966.2 | 1957.0 | 1700.5 |

Through the public entry it is slower than Triton in 11 of 12 cells. Called directly, it wins at
small batch and loses at b6 s8 and b32. Two follow-ups would make it worth merging: a wave32
tile/rung picker, and cutting the public entry's per-call overhead (about 30-40 us at small
batch). Its production value also needs a vLLM change, because vLLM does not call it on RDNA4.

### 3.6 FlyDSL `causal_conv1d_update`: one bit-exactness assertion fails

`test_dispatch_seam_routes_to_flydsl[8-256-4-1-False-False-True]` asserts that the 2D decode
output of the FlyDSL route is bit-identical to Triton's. Its helper's message ("assembled from
copies, never computed") holds for the conv state but not for the output, which is computed.
On gfx1201 the two kernels round differently: 1,049 of 2,048 elements differ, max 0.0625.
Against an fp64 oracle FlyDSL is the more accurate (max 0.031 / mean 0.0019 against Triton's
0.039 / 0.0027). The test was left failing. The likely fix is to compare the 2D output with the
oracle, as the 3D cases already do, but that is the test author's call. The route is opt-in
(`AITER_CONV1D_UPDATE_FLYDSL`). It is slower than the best baseline mostly at batch 128 verify
(up to x1.55).

### 3.7 qk_norm_rope_quant has no non-torch baseline

The gfx1201 route fixes a compile failure (the wave64 builder needs gfx950's
`permlane16.swap`): 46 failing checks become 34 pass and 12 warn. There is nothing faster to
compare it with, because before the change the op did not compile on gfx1201 at all.

### 3.8 Coverage and environment limits

- `test_fused_qk_norm_mrope_cache_quant.py` is not in the Phase 1 baseline table; its first run
  on gfx1201 was in Phase 2 (190 checks, none fail).
- Six `test_qsa.py` failures assert on kernel names in a torch profiler trace, and this image's
  profiler records no device kernels. They are an environment limit, not a code failure.
- Several default test shapes need more than 32 GB and fail with `torch.OutOfMemoryError`: the
  topk 90,000-row prefill case, the MoE a8w8 M=4096 / N=7168 / K=4096 / 256-expert cases
  (14 GiB in one allocation, even with `expandable_segments`), and the largest qk_norm_rope
  reference.
