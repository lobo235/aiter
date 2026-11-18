# aiter on gfx1201 (Radeon AI PRO R9700): Phase 1 status

This report covers Phase 1 of the RDNA4 work: a baseline, one fix and two read-only
audits. Only the LDS fix changes code (see section 2). Every failure in section 1 is
a finding. No test was skipped, loosened or turned off to make it pass.

Base commit: `ca7cfa1c3` (ROCm/aiter `main`). Branch: `gfx1201-phase1`.

Two assumptions shape the ranking:

- **Model.** The brief left the target model list empty. The impact columns use
  **Qwen3.8-27B**: hidden 5120, intermediate 17408, 64 layers (48 Gated-DeltaNet and
  16 full-attention), 24 q / 4 kv heads, head_dim 256, vocab 248320. It is dense, with
  no MoE or MLA. A different model list changes sections 3 to 5, not sections 1 and 2.
- **Consumer.** "Called by vLLM" means vLLM main as of 2026-11-10 (`b6625e2421`) and a
  vLLM 0.31 ROCm build. Both were read from source; neither was run.

## 1. Environment and baseline

### Environment

| item | value |
|---|---|
| GPU | AMD Radeon AI PRO R9700, `gfx1201` (from `rocminfo \| grep gfx`) |
| Compute units | 64 (rocminfo). torch reports `multi_processor_count` 32, which is the WGP count |
| Wave size | 32 |
| LDS per workgroup | 64 KiB. Three sources agree: rocminfo `GROUP` segment `64(0x40) KB`, torch `shared_memory_per_block` 65536, Triton `max_shared_mem` 65536 |
| Software | a vLLM ROCm container image with Python 3.12.14, torch 2.13.0, HIP 7.15, Triton 3.8.0 |
| FlyDSL | 0.5.0. The image has none, so it was installed from PyPI into a side directory on `PYTHONPATH` |
| Build | `GPU_ARCHS=gfx1201`, `AITER_ROCM_ARCH=gfx1201`, `PYTORCH_ROCM_ARCH=gfx1201`. The JIT and Triton caches were private to these runs, and the repo was mounted at a separate path so the image's own aiter copy was not used |

The tests ran with `pytest -q`, one process per file and a 1 h limit per file
(30 min for `opus/device`).

**Scope.** The brief named these suites:

- `op_tests/triton_tests/` `gemm`, `normalization`, `rope`, `quant`, `attention` and
  `conv`, plus `moe/test_moe_gemm_a8w8.py` and `moe/test_moe_gemm_a8w8_blockscale.py`
- `op_tests/test_flydsl_fmha.py`
- `op_tests/opus/device`

pytest cannot collect `op_tests/opus/device/test_opus_device.py`: its test functions
take a `mod` argument that is not a fixture. It was also run as a script, the way its
`__main__` block intends, and both results are shown below.

### Results by area

The full per-file table is at the end of this section.

| area | files | pass | fail | skip | files that did not finish |
|---|---:|---:|---:|---:|---:|
| `triton_tests/gemm` | 29 | 1647 | 5524 | 15908 | 1 |
| `triton_tests/normalization` | 6 | 1438 | 170 | 420 | 0 |
| `triton_tests/rope` | 5 | 1546 | 2091 | 278 | 0 |
| `triton_tests/quant` | 15 | 1208 | 0 | 3186 | 0 |
| `triton_tests/attention` | 45 | 6155 | 587 | 10038 | 1 |
| `triton_tests/conv` | 4 | 311 | 0 | 0 | 1 |
| `triton_tests/moe` (the two files above) | 2 | 384 | 384 | 1792 | 0 |
| `test_flydsl_fmha.py` | 1 | 60 | 0 | 260 | 0 |
| `opus/device` under pytest | 1 | 0 | 80 | 0 | 0 |
| `opus/device` as a script | 1 | 62 | 0 | 49 | 0 |
| **total under pytest** | **108** | **12749** | **8836** | **31882** | **3** |

The 31,882 skips are the tests' own arch gates, and they are expected. The largest
groups are:

| skip reason | cases |
|---|---:|
| MXFP4 | 12785 |
| Gluon backends that exist only for gfx950 or gfx1250 | about 7,000 |
| gfx950-only configs | 1694 |
| CDNA4-only float8 x MX | 1152 |
| FNUZ FP8 (gfx942) | 640 |
| `gemm_a16w4 has no config for gfx1201` | 364 |

### Why the failing cases fail

| cause | failing cases | where |
|---|---:|---|
| **Missing gfx1201 config.** The loader raises `Required config file doesn't exist` with no fallback; section 4 explains this | 5844 | 15 GEMM files, `normalization` (`rmsnorm`, `fused_reduce_add_rmsnorm`), `attention` (`mla_decode_rope`, `extend_attention`, `hstu_attn`), grouped GMM |
| **Numerical mismatch** | 2094 | 2091 in `rope/test_rope.py` (841) and `rope/test_fused_qkv_split_qk_rope.py` (1250); 2 in `test_mha_with_sink.py`; 1 in `test_rmsnorm.py` |
| **Out of memory.** The test shapes need more than the card's 32 GB | 791 | `test_unified_attention`, `test_mha*`, `test_moe_gemm_a8w8_blockscale` (allocations of 7 to 28 GiB), `test_gmm`, `test_ff_a16w16`, and others |
| **pytest cannot collect `test_opus_device.py`** (described above) | 80 | `opus/device`. All 62 runnable cases pass as a script |
| **JIT build failure: `module_top_k_per_row`** | 21 | `test_qsa.py` |
| **Kernel-name assertions** | 6 | `test_qsa.py` |

Details for the failure causes:

- **Rope numerics.** A typical failure has 3 elements out of 64 or 512 off by one bf16
  ULP: absolute differences of 0.0039 and 0.0078 against tolerances of 1e-5 and 1e-3.
  The cause was not investigated. Phase 2 must decide whether the kernel or the
  reference is at fault before anyone touches a tolerance.
- **`module_top_k_per_row` build failure.**
  `csrc/kernels/topk_per_row_kernels.cu:4443` needs 66,824 B of local memory for
  `radix_topk_one_block_kernel<float, int, 12, 1024, ...>`. The limit is 65,536 B, so
  the module does not compile for gfx1201.
- **Kernel-name assertions.** In this image the torch profiler records HIP API calls
  but no kernel events. That causes these failures, and it also makes every
  `run_perftest` time read 0 µs (see "Measurement" below).

Three files did not finish:

| file | what happened |
|---|---|
| `attention/test_sol_tilt.py` | aborts the process (SIGABRT) inside the JIT op `rotate_activation_hd128` (`aiter/ops/mha_v4_quant.py:99`) |
| `conv/test_causal_conv1d.py` | aborts at 62 % of its cases, inside the **torch reference** (`F.conv1d`, test line 47), not in aiter |
| `gemm/batched/test_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py` | the host's out-of-memory guard killed it (SIGTERM) at about 40 GiB of resident memory, a third of the way through its cases. This happened in both runs, after about 23 min each time. Host memory grows across cases in one process |

### The ungated FlyDSL GDN decode/verify kernel

The gate audit (section 3) found that `aiter/ops/flydsl/kernels/gdn_decode_verify.py`
can be reached on gfx1201. It reduces with `shuffle_xor(..., 64)`, and its test skips
every arch except gfx942 and gfx950.

It was run through the test's own helper, `test_flydsl_gdn_decode_varlen`, unchanged:
same reference, same inputs, same tolerance. Only the arch skip was left out. Inputs
were 16 k-heads, 48 v-heads, head_v_dim 128 and bf16 gates, over 6 (batch, tokens per
sequence) pairs × 3 (cache_states, update_state) modes. **All 18 cases pass**:
output, state pool and snapshots all match the reference at 1e-2.

So at the default `KSPLIT` the width-64 shuffle gives correct results on wave32.
vLLM does not call this kernel on RDNA4 today.

### Measurement

**`run_perftest` cannot time anything in this image.** The profiler post-processing
reports `no valida data after post process!`, and the timing fields read 0. Phase 2
needs a working timer before the rule "a port slower than the current fallback is a
failure" can be applied (section 5, step 0).

### Corrections to the brief

The brief's directory counts are off:

| directory | brief | actual |
|---|---:|---:|
| `configs/gfx1201/triton/gemm/` | 4 families | 6 |
| `configs/gfx1250/triton/gemm/` | 24 | 35 families plus `.gitkeep` |

<details>
<summary>Per-file baseline (108 files)</summary>

| test file | pass | fail | skip | note |
|---|---:|---:|---:|---|
| `triton_tests/gemm/basic/test_gemm_a16w16.py` | 5 | 372 | 764 |  |
| `triton_tests/gemm/basic/test_gemm_a16w16_gated.py` | 0 | 900 | 1692 |  |
| `triton_tests/gemm/basic/test_gemm_a16w4.py` | 0 | 0 | 364 |  |
| `triton_tests/gemm/basic/test_gemm_a16w8_blockscale.py` | 0 | 46 | 2 |  |
| `triton_tests/gemm/basic/test_gemm_a16wfp4.py` | 0 | 0 | 104 |  |
| `triton_tests/gemm/basic/test_gemm_a8w8.py` | 753 | 712 | 380 |  |
| `triton_tests/gemm/basic/test_gemm_a8w8_blockscale.py` | 233 | 0 | 248 |  |
| `triton_tests/gemm/basic/test_gemm_a8w8_per_token_scale.py` | 0 | 136 | 0 |  |
| `triton_tests/gemm/basic/test_gemm_a8wfp4.py` | 0 | 0 | 23 |  |
| `triton_tests/gemm/basic/test_gemm_afp4wfp4.py` | 0 | 0 | 632 |  |
| `triton_tests/gemm/basic/test_gemm_afp8wfp8.py` | 0 | 320 | 519 |  |
| `triton_tests/gemm/batched/test_batched_gemm_a16w8.py` | 0 | 0 | 170 |  |
| `triton_tests/gemm/batched/test_batched_gemm_a16wfp4.py` | 1 | 0 | 2328 |  |
| `triton_tests/gemm/batched/test_batched_gemm_a8w8.py` | 0 | 196 | 0 |  |
| `triton_tests/gemm/batched/test_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py` | 0 | 0 | 0 | killed by the host out-of-memory guard (SIGTERM) |
| `triton_tests/gemm/batched/test_batched_gemm_afp4wfp4.py` | 0 | 0 | 2668 |  |
| `triton_tests/gemm/batched/test_batched_gemm_bf16.py` | 0 | 240 | 720 |  |
| `triton_tests/gemm/feed_forward/test_ff_a16w16.py` | 0 | 864 | 0 |  |
| `triton_tests/gemm/feed_forward/test_ff_a16w16_fused.py` | 0 | 736 | 132 |  |
| `triton_tests/gemm/fused/test_fused_gemm_a16w16_dual_splitk.py` | 0 | 0 | 1694 |  |
| `triton_tests/gemm/fused/test_fused_gemm_a16w16_mul_add.py` | 0 | 216 | 0 |  |
| `triton_tests/gemm/fused/test_fused_gemm_a16w16_quant_x.py` | 0 | 84 | 0 |  |
| `triton_tests/gemm/fused/test_fused_gemm_a8w8_blockscale_a16w16.py` | 0 | 584 | 0 |  |
| `triton_tests/gemm/fused/test_fused_gemm_a8w8_blockscale_mul_add.py` | 642 | 0 | 0 |  |
| `triton_tests/gemm/fused/test_fused_gemm_a8w8_blockscale_split_cat.py` | 12 | 0 | 0 |  |
| `triton_tests/gemm/fused/test_fused_gemm_afp4wfp4_a16w16.py` | 0 | 0 | 1152 |  |
| `triton_tests/gemm/fused/test_fused_gemm_afp4wfp4_mul_add.py` | 0 | 0 | 2048 |  |
| `triton_tests/gemm/fused/test_fused_gemm_afp4wfp4_split_cat.py` | 0 | 0 | 208 |  |
| `triton_tests/gemm/grouped/test_gmm.py` | 1 | 118 | 60 |  |
| `triton_tests/normalization/test_fused_add_rmsnorm_pad.py` | 840 | 0 | 420 |  |
| `triton_tests/normalization/test_fused_reduce_add_rmsnorm.py` | 0 | 128 | 0 |  |
| `triton_tests/normalization/test_fused_rmsnorm_add.py` | 32 | 0 | 0 |  |
| `triton_tests/normalization/test_fused_rmsnorm_add_rmsnorm.py` | 12 | 0 | 0 |  |
| `triton_tests/normalization/test_layernorm.py` | 144 | 0 | 0 |  |
| `triton_tests/normalization/test_rmsnorm.py` | 410 | 42 | 0 |  |
| `triton_tests/rope/test_fused_qk_norm_rope_cached.py` | 17 | 0 | 0 |  |
| `triton_tests/rope/test_fused_qk_norm_rope_gate_fp8_quant.py` | 6 | 0 | 20 |  |
| `triton_tests/rope/test_fused_qkv_split_qk_rope.py` | 775 | 1250 | 256 |  |
| `triton_tests/rope/test_rope.py` | 726 | 841 | 0 |  |
| `triton_tests/rope/test_rope_mxfp4.py` | 22 | 0 | 2 |  |
| `triton_tests/quant/test_convert_mxfp8.py` | 31 | 0 | 4 |  |
| `triton_tests/quant/test_dequant_hadamard_quant_mxfp4.py` | 0 | 0 | 32 |  |
| `triton_tests/quant/test_fast_transpose.py` | 48 | 0 | 0 |  |
| `triton_tests/quant/test_fused_fp8_quant.py` | 490 | 0 | 0 |  |
| `triton_tests/quant/test_fused_mxfp4_quant.py` | 0 | 0 | 3051 |  |
| `triton_tests/quant/test_fused_quant_transpose_hadamard_mxfp4.py` | 10 | 0 | 20 |  |
| `triton_tests/quant/test_fused_rms_gated_fp8_group_quant.py` | 121 | 0 | 0 |  |
| `triton_tests/quant/test_mxfp4_jit_import.py` | 60 | 0 | 0 |  |
| `triton_tests/quant/test_mxfp4_philox.py` | 18 | 0 | 0 |  |
| `triton_tests/quant/test_quant.py` | 138 | 0 | 0 |  |
| `triton_tests/quant/test_quant_configs.py` | 24 | 0 | 0 |  |
| `triton_tests/quant/test_quant_fp8_blockwise.py` | 22 | 0 | 0 |  |
| `triton_tests/quant/test_quant_mxfp4.py` | 62 | 0 | 71 |  |
| `triton_tests/quant/test_quant_mxfp4_blockscale.py` | 29 | 0 | 0 |  |
| `triton_tests/quant/test_quant_mxfp8.py` | 155 | 0 | 8 |  |
| `triton_tests/attention/test_block_scale_fp8_attention.py` | 37 | 0 | 2 |  |
| `triton_tests/attention/test_chunk_kda.py` | 0 | 0 | 63 |  |
| `triton_tests/attention/test_chunked_pa_prefill.py` | 16 | 0 | 0 |  |
| `triton_tests/attention/test_dsv4_indexer.py` | 3 | 0 | 0 |  |
| `triton_tests/attention/test_extend_attention.py` | 0 | 32 | 0 |  |
| `triton_tests/attention/test_fav3_sage.py` | 525 | 0 | 584 |  |
| `triton_tests/attention/test_fav3_sage_compile.py` | 17 | 0 | 0 |  |
| `triton_tests/attention/test_fp8_mqa_logits.py` | 253 | 0 | 0 |  |
| `triton_tests/attention/test_fused_attention_qk_norm_rope.py` | 0 | 0 | 10 |  |
| `triton_tests/attention/test_hstu_attention_ref.py` | 29 | 0 | 0 |  |
| `triton_tests/attention/test_hstu_attn.py` | 0 | 2 | 0 |  |
| `triton_tests/attention/test_kda.py` | 0 | 0 | 276 |  |
| `triton_tests/attention/test_la_paged.py` | 17 | 0 | 0 |  |
| `triton_tests/attention/test_lse_merge.py` | 87 | 0 | 0 |  |
| `triton_tests/attention/test_mha.py` | 1128 | 73 | 1192 |  |
| `triton_tests/attention/test_mha_dao_ai.py` | 52 | 0 | 0 |  |
| `triton_tests/attention/test_mha_fp8.py` | 124 | 0 | 52 |  |
| `triton_tests/attention/test_mha_fp8_e2e.py` | 0 | 0 | 204 |  |
| `triton_tests/attention/test_mha_mxfp8.py` | 0 | 0 | 36 |  |
| `triton_tests/attention/test_mha_spatial_swizzle.py` | 22 | 0 | 0 |  |
| `triton_tests/attention/test_mha_v3.py` | 690 | 0 | 0 |  |
| `triton_tests/attention/test_mha_with_pe.py` | 220 | 12 | 164 |  |
| `triton_tests/attention/test_mha_with_sink.py` | 328 | 2 | 360 |  |
| `triton_tests/attention/test_mla.py` | 26 | 0 | 779 |  |
| `triton_tests/attention/test_mla_decode_rope.py` | 0 | 240 | 0 |  |
| `triton_tests/attention/test_mla_v4_triton.py` | 96 | 0 | 24 |  |
| `triton_tests/attention/test_mxfp8_attention_kernel.py` | 0 | 0 | 23 |  |
| `triton_tests/attention/test_na3d_flash.py` | 0 | 0 | 20 |  |
| `triton_tests/attention/test_pa_decode.py` | 938 | 0 | 216 |  |
| `triton_tests/attention/test_pa_decode_sparse.py` | 100 | 0 | 347 |  |
| `triton_tests/attention/test_pa_mqa_logits_2d_context_lens.py` | 0 | 0 | 6 |  |
| `triton_tests/attention/test_pa_mqa_logits_mxfp4.py` | 0 | 0 | 536 |  |
| `triton_tests/attention/test_pa_prefill.py` | 18 | 0 | 0 |  |
| `triton_tests/attention/test_pa_prefill_sparse.py` | 16 | 0 | 32 |  |
| `triton_tests/attention/test_paged_attention_output_gate.py` | 0 | 0 | 501 |  |
| `triton_tests/attention/test_prefill_attention.py` | 26 | 6 | 0 |  |
| `triton_tests/attention/test_qsa.py` | 83 | 27 | 0 |  |
| `triton_tests/attention/test_sol_prepare.py` | 87 | 0 | 4 |  |
| `triton_tests/attention/test_sol_tilt.py` | 0 | 0 | 0 | process aborted |
| `triton_tests/attention/test_sparse_attention_dsv4_bwd.py` | 0 | 0 | 8 |  |
| `triton_tests/attention/test_sparse_mla.py` | 27 | 0 | 117 |  |
| `triton_tests/attention/test_sparse_mla_dsv4_train.py` | 18 | 0 | 0 |  |
| `triton_tests/attention/test_unified_attention.py` | 1028 | 193 | 4476 |  |
| `triton_tests/attention/test_unified_attention_prefill_recovery.py` | 0 | 0 | 6 |  |
| `triton_tests/attention/test_unified_attention_sparse_mla.py` | 144 | 0 | 0 |  |
| `triton_tests/conv/test_causal_conv1d.py` | 0 | 0 | 0 | process aborted |
| `triton_tests/conv/test_causal_conv1d_update_single_token.py` | 26 | 0 | 0 |  |
| `triton_tests/conv/test_conv2d.py` | 133 | 0 | 0 |  |
| `triton_tests/conv/test_conv3d.py` | 152 | 0 | 0 |  |
| `triton_tests/moe/test_moe_gemm_a8w8.py` | 0 | 0 | 1792 |  |
| `triton_tests/moe/test_moe_gemm_a8w8_blockscale.py` | 384 | 384 | 0 |  |
| `test_flydsl_fmha.py` | 60 | 0 | 260 |  |
| `opus/device/test_opus_device.py` | 0 | 80 | 0 |  |
| **total (108 files)** | **12749** | **8836** | **31882** | |

</details>

## 2. LDS cap fix

### The fix

`pick_gemm_num_stages()` (`aiter/ops/triton/utils/gemm_config_utils.py`) reads
`arch_info._LDS_CAP_BYTES`. gfx1200 and gfx1201 were missing from that table, so for
them the function returned 2 stages without checking the tiles against LDS. The JIT's
own table, `aiter/jit/utils/lds_capacity.py`, already had 64 KiB for both. The
device-property check in section 1 confirms 64 KiB.

```diff
--- a/aiter/ops/triton/utils/_triton/arch_info.py
+++ b/aiter/ops/triton/utils/_triton/arch_info.py
@@ -83,4 +83,12 @@ def is_cdna4():
     return get_arch() == "gfx950"


-_LDS_CAP_BYTES = {"gfx1250": 327680, "gfx950": 163840, "gfx942": 65536}
+# Bytes of LDS one workgroup can allocate. Keep in step with
+# aiter/jit/utils/lds_capacity.py, which holds the same limit for the JIT.
+_LDS_CAP_BYTES = {
+    "gfx1250": 327680,
+    "gfx950": 163840,
+    "gfx942": 65536,
+    "gfx1200": 65536,
+    "gfx1201": 65536,
+}
```

### The new test

`op_tests/triton_tests/utils/test_lds_cap_gfx12.py` has two checks:

- the Triton and JIT tables agree at 64 KiB for gfx1200 and gfx1201;
- `pick_gemm_num_stages` keeps two stages when they fit and drops to one when they
  don't, on both arches.

| tile (BM×BN×BK) | data | stages expected |
|---|---|---:|
| 256×256×64 | bf16 | 2 |
| 256×256×128 | bf16 | 1 |
| 64×128×128 | fp8, async-padded | 2 |
| 128×128×128 | fp8, async-padded | 1 |
| 256×256×256 | fp8, async-padded | 1 |

The test does not need a GPU.

### Before and after

| run | `test_lds_cap_gfx12.py` |
|---|---|
| before the fix | 4 pass, **8 fail**: both table checks fail on the missing key, and all six one-stage cases return 2 |
| after the fix | 12 pass |

The 4 that pass before the fix are the two-stage cases. The old code returned 2 for
every tile, so they pass by accident.

Every test that could reach the changed code was run again after the fix, on the same
image, caches and card:

- **The GEMM and MoE files** are compared with the section 1 baseline.
- **The new test and four extra files** (`moe_gemm_a4w4`, `moe_gemm_a8w4`, `sparse_mla`,
  `pa_decode_sparse`) were first run on the unfixed tree in the same container.

Every file matches its before result exactly, except `test_lds_cap_gfx12.py`:

| test file | pass | fail | skip | note |
|---|---:|---:|---:|---|
| `triton_tests/utils/test_lds_cap_gfx12.py` | 12 | 0 | 0 |  |
| `triton_tests/moe/test_moe_gemm_a4w4.py` | 0 | 0 | 672 |  |
| `triton_tests/moe/test_moe_gemm_a8w4.py` | 0 | 0 | 5632 |  |
| `triton_tests/attention/test_sparse_mla.py` | 27 | 0 | 117 |  |
| `triton_tests/attention/test_pa_decode_sparse.py` | 100 | 0 | 347 |  |
| `triton_tests/moe/test_moe_gemm_a8w8.py` | 0 | 0 | 1792 |  |
| `triton_tests/moe/test_moe_gemm_a8w8_blockscale.py` | 384 | 384 | 0 |  |
| `triton_tests/gemm/basic/test_gemm_a16w16.py` | 5 | 372 | 764 |  |
| `triton_tests/gemm/basic/test_gemm_a16w16_gated.py` | 0 | 900 | 1692 |  |
| `triton_tests/gemm/basic/test_gemm_a16w4.py` | 0 | 0 | 364 |  |
| `triton_tests/gemm/basic/test_gemm_a16w8_blockscale.py` | 0 | 46 | 2 |  |
| `triton_tests/gemm/basic/test_gemm_a16wfp4.py` | 0 | 0 | 104 |  |
| `triton_tests/gemm/basic/test_gemm_a8w8.py` | 753 | 712 | 380 |  |
| `triton_tests/gemm/basic/test_gemm_a8w8_blockscale.py` | 233 | 0 | 248 |  |
| `triton_tests/gemm/basic/test_gemm_a8w8_per_token_scale.py` | 0 | 136 | 0 |  |
| `triton_tests/gemm/basic/test_gemm_a8wfp4.py` | 0 | 0 | 23 |  |
| `triton_tests/gemm/basic/test_gemm_afp4wfp4.py` | 0 | 0 | 632 |  |
| `triton_tests/gemm/basic/test_gemm_afp8wfp8.py` | 0 | 320 | 519 |  |
| `triton_tests/gemm/batched/test_batched_gemm_a16w8.py` | 0 | 0 | 170 |  |
| `triton_tests/gemm/batched/test_batched_gemm_a16wfp4.py` | 1 | 0 | 2328 |  |
| `triton_tests/gemm/batched/test_batched_gemm_a8w8.py` | 0 | 196 | 0 |  |
| `triton_tests/gemm/batched/test_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py` | 0 | 0 | 0 | killed by the host out-of-memory guard (SIGTERM) |
| `triton_tests/gemm/batched/test_batched_gemm_afp4wfp4.py` | 0 | 0 | 2668 |  |
| `triton_tests/gemm/batched/test_batched_gemm_bf16.py` | 0 | 240 | 720 |  |
| `triton_tests/gemm/feed_forward/test_ff_a16w16.py` | 0 | 864 | 0 |  |
| `triton_tests/gemm/feed_forward/test_ff_a16w16_fused.py` | 0 | 736 | 132 |  |
| `triton_tests/gemm/fused/test_fused_gemm_a16w16_dual_splitk.py` | 0 | 0 | 1694 |  |
| `triton_tests/gemm/fused/test_fused_gemm_a16w16_mul_add.py` | 0 | 216 | 0 |  |
| `triton_tests/gemm/fused/test_fused_gemm_a16w16_quant_x.py` | 0 | 84 | 0 |  |
| `triton_tests/gemm/fused/test_fused_gemm_a8w8_blockscale_a16w16.py` | 0 | 584 | 0 |  |
| `triton_tests/gemm/fused/test_fused_gemm_a8w8_blockscale_mul_add.py` | 642 | 0 | 0 |  |
| `triton_tests/gemm/fused/test_fused_gemm_a8w8_blockscale_split_cat.py` | 12 | 0 | 0 |  |
| `triton_tests/gemm/fused/test_fused_gemm_afp4wfp4_a16w16.py` | 0 | 0 | 1152 |  |
| `triton_tests/gemm/fused/test_fused_gemm_afp4wfp4_mul_add.py` | 0 | 0 | 2048 |  |
| `triton_tests/gemm/fused/test_fused_gemm_afp4wfp4_split_cat.py` | 0 | 0 | 208 |  |
| `triton_tests/gemm/grouped/test_gmm.py` | 1 | 118 | 60 |  |
| **total (36 files)** | **2170** | **5908** | **24468** | |

### What the fix does and does not change

- **Dense GEMMs: nothing.** Every dense GEMM family takes `num_stages` from its JSON
  config. Only the three MoE GEMMs call `pick_gemm_num_stages`:
  - `moe/moe_op_gemm_a8w8.py:141`
  - `moe/moe_op_gemm_a8w4.py:242`
  - `moe/moe_op_gemm_a4w4.py:169`
- **No existing test reaches the changed path on gfx1201.**
  - `test_moe_gemm_a8w8.py` tests only the MX and FNUZ variants, and both skip here.
  - The a8w4 and a4w4 tests need FP4.

  An OCP-FP8 MoE GEMM case for RDNA4 does not exist in the suite. It is listed in
  section 5.
- **The direct table lookups were already unreachable on gfx1201.**
  `moe_op_gemm_a4w4.py:51`, `attention/sparse_mla.py:176` and
  `attention/pa_decode_sparse.py:498` / `:1446` index the table directly. Each is
  guarded so it cannot run on gfx1201: the a4w4 kernel needs FP4,
  `_check_lds_budget` returns early for any arch but gfx942, and the pa_decode_sparse
  lines run only on the Gluon path.

  So the fix removes no reachable KeyError. The gate audit's claim that it did is
  corrected here.
- **The tuner.** It already used `_LDS_CAP_BYTES.get(arch, 64 * 1024)`, so sweeps on
  gfx1201 do not change.

## 3. Gate audit

**Method.** The audit read 524 Python lines and 418 `csrc` preprocessor lines that
mention an arch, plus the JIT and build tables. Instruction support was checked by
assembling probes with `llvm-mc -mcpu=gfx1201` and lowering intrinsics with
`llc -mcpu=gfx1201` (LLVM 21.1.8). Nothing was changed.

**What does not block gfx1201:**

- The JIT does not exclude gfx12. `aiter/jit/core.py:960-994` accepts gfx1200 and
  gfx1201, and `optCompilerConfig.json` has no per-module arch list.
- One gfx1201 HIP kernel already compiles in wave64 mode: `module_chunk_gdr_fwd_h` with
  `-mwavefrontsize64`. Some wave64 HIP code can therefore be ported by compile flag
  rather than by rewrite. FlyDSL and Triton pick 32 lanes on gfx12.

**Already enabled for gfx1201** (no action):

- the Triton GDN chunk prefill
- the HIP `chunk_gated_delta_rule_fwd_h`
- FlyDSL `fmha_gfx1201` and `fp8_quant_gfx1201`
- gfx12 WMMA in `opus.hpp`
- the OCP FP8 helpers
- Triton `unified_attention`

### Possibly portable

The guarded code uses only instructions gfx1201 has. What blocks it is the gate, a
wave-size constant or a missing config.

| file:line | what is gated | work needed | wave64 risk | Qwen3.8-27B impact |
|---|---|---|---|---|
| `aiter/ops/flydsl/linear_attention_kernels.py:454`, kernel `kernels/gdr_decode.py` | FlyDSL GDN/GDR decode recurrence and its MTP verify path. VALU and shuffles, no MMA | Derive the lane math from `get_warp_size()` (`gdr_decode.py:57/77/633/648`, host tiling at `linear_attention_kernels.py:104/111/243/251`), and add gfx1201 tuning entries | **high**: wave64 is hard-coded, and a mismatch corrupts output silently | very high in principle (48 layers, every decode and verify step). vLLM does not call it on RDNA4 today |
| `aiter/ops/flydsl/kernels/gdn_decode_verify.py` (no gate) | `flydsl_gdn_decode_varlen` | none. **Verified correct on gfx1201** (section 1, 18/18) | measured: fine at the default `KSPLIT` | same class as above; not called by vLLM on RDNA4 |
| `aiter/ops/flydsl/causal_conv1d_update_kernels.py:75` | FlyDSL `causal_conv1d_update`: one thread per channel, no shuffles | Add the arch and validate | low | moderate. A Triton alternative exists |
| `aiter/ops/triton/gated_delta_net/fused_gdn_decode_qkvz.py:95-115` | Gluon fused GDN decode: no MFMA, cdna3 buffer ops | Rewrite the layouts for 32 lanes. Gluon buffer ops on gfx12 are unverified, and the tiles are fixed to one GDN shape | high | high if the shapes match |
| `csrc/kernels/topk_per_row_kernels.cu:4443` | not an arch gate: the module fails to compile (section 1) | Fit `radix_topk_one_block_kernel` (12-bit, 1024 threads) into 64 KiB, or skip that instance on 64 KiB parts | — | none for Qwen. Used by the QSA/MQA top-k paths |
| `csrc/kernels/rope/rope_common.h:7404` | packed FP8 convert in `vec_t::from_` | Add gfx1200/1201 to the `fp8e4m3fn` branch only. The scalar fallback is correct today | none | small, and only with an FP8 KV cache |
| `csrc/include/opus/opus.hpp:1780/1785/1790` | native packed f16/bf16 atomic add and f32 atomic min/max | Add `__gfx1200__ \|\| __gfx1201__`. The lowering was verified with `llc` | none | low (split-K reductions in opus) |
| `aiter/ops/flydsl/kernels/qk_norm_rope_quant.py:1448` | picks the wave64 builder on every arch except gfx1250 | Pick the existing `_build_kernel_w32` when the warp size is 32 | medium: it uses the wrong builder today | low (GPT-J RoPE; Qwen uses NeoX) |
| `aiter/ops/triton/utils/_triton/tuning/space.py:219` | the tuner's register-spill filter assumes wave64 on every arch except gfx1250 | One line covering gfx10/11/12 | it **is** a wave-size bug | none at run time. It lets about 5 % spill-only tiles into every gfx1201 sweep |
| `aiter/tuned_gemm.py:~317` | the Triton fallback for untuned BF16 shapes exists only for gfx1250 | Add gfx1200/1201 once `gemm_a16w16` has a config (section 4) | none | low for a quantized 27B |
| `csrc/kernels/custom_kernels.cu:21-30` | wvSplitK skinny GEMM | A real wave32 rewrite. DPP `row_bcast:15/31` and `v_dot2c_f32_f16` are rejected on gfx1201 | high | low for a quantized 27B |
| `aiter/ops/triton/normalization/rmsnorm.py:34/58/221` | gfx950 tuning heuristics | Add gfx1201 values | none | small |
| `aiter/ops/flydsl/mla_reduce_kernels.py:43`, `aiter/ops/triton/attention/na3d_flash.py:30`, `aiter/aot/flydsl/*` | MLA reduce, 3D neighbourhood attention, AOT lists | Port or add the arch | varies | none for Qwen |

### Needs a rewrite

These depend on features gfx1201 does not have.

| area | depends on |
|---|---|
| `_gluon_kernels/gfx1250/**` (39 files) and FlyDSL `*_gfx1250` (31 files) | TDM, clusters, gfx1250 WMMA shapes, MX / WMMA-scale |
| `qk_norm_rope_quant.py:1780` (TDM prefill), `jit/core.py:1401-1411` (cluster launch), MHC / all-reduce gfx1250 modules | TDM, clusters, gfx1250 comms |
| `aiter_opus_plus.h:106` and `:646-1039`, `inverse_rope_group_quant.cu:150-185` | `v_cvt_scalef32_*`, which gfx1201 rejects. Plain-convert fallbacks already serve gfx1201 |
| FP4 / MX on gfx950 (`is_fp4_avail`, `quant_mxfp4.cu`, the MXFP4 attention and GEMM paths) | FP4 / MX hardware, which gfx1201 lacks |

### CDNA-only and correct

Each of these guards MFMA, wave64-only DPP, packed FP32, `buffer_load ... lds`, XGMI,
prebuilt CDNA code objects or an FNUZ encoding. gfx1201 already gets a correct
fallback or another path.

- MHA / FMHA v3 ASM, PA ASM, MLA ASM and metadata
- HIP paged attention (MFMA). Triton `unified_attention` covers gfx1201
- FlyDSL `pa_decode` and GDN prefill (MFMA). Triton and HIP prefill cover gfx1201
- gfx950 Gluon kernels
- packed-FP32 asm with a C++ fallback
- 64-bit-mask `v_cmp` asm with a portable `#else`
- FNUZ-vs-OCP selections, correct for OCP gfx1201
- MoE (`fused_moe`, FlyDSL MoE): not used by a dense model
- XGMI comms
- the DeepSeek MLA fusions
- prebuilt / CK GEMMs

## 4. Tuning gaps, ranked

### How a missing gfx1201 config fails

`get_gemm_config()` resolves `configs/<arch>/<backend>/gemm/<family>/`. It never
tries another arch, another backend or a CU-count fallback. A family without a
`DEFAULT.json` raises `AssertionError("Required config file doesn't exist")` at its
first call; `configs/AGENTS.md` rule 1 forbids Python fallbacks. Grouped GMM reads the
Gluon tree only and raises `FileNotFoundError`.

That is the single largest cause in section 1: 5,844 failing cases. The tuner also
cannot fix it on its own: it learns a family's keys from `DEFAULT.json` and never
writes one.

There are two kinds of fix:

- **A missing family is a crash.** The fix is one `DEFAULT.json`, seeded by hand.
- **A missing shape inside a tuned family is a speed gap.** Only a sweep fixes it.

| rank | gap | kind | Qwen3.8-27B | general | evidence |
|---:|---|---|---|---|---|
| 1 | `gemm_a8w8_blockscale` shape files for K = 5120 / 6144 / 17408 | shape coverage | high **for an FP8-blockscale checkpoint**. vLLM's only aiter GEMM route on RDNA is blockscale, gated per (N, K) on `is_tuned` probed at M = 128. None of the 31 gfx1201 files has K = 5120, so no Qwen layer takes it | high for FP8-blockscale releases | source (vLLM `aiter.py:456-492`) and a directory listing |
| 2 | `gemm_a16w16` and its BF16 siblings (`_persistent`, `_atomic`, `_gated`, `ff_a16w16_fused`, `fused_gemm_a16w16_mul_add`, `batched_gemm_a16w16`) | missing families | medium (lm_head and unquantized layers). vLLM does not route BF16 to aiter on RDNA | **high**: the default format, and a crash. gfx1100/1101/1150/1151 all have it | measured: `test_gemm_a16w16.py` 372 fail / 5 pass |
| 3 | `gemm_a16w4` (int4 W4A16) | missing family | high **if** a framework calls it | high: the common quant on 16 to 32 GB cards | test skips (`no config for gfx1201`) |
| 4 | `gemm_a8w8` Qwen shapes | shape coverage | low (vLLM's per-token FP8 aiter path is gfx9 only) | medium (Llama3-8B is covered) | directory listing |
| 5 | `gemm_a16w8_blockscale` (+ preshuffled) | missing family | medium | medium | measured: 46 fail |
| 6 | `gemm_a8w8_per_token_scale` (INT8), `gemm_a8w8_preshuffled`, `batched_gemm_a8w8`, fused blockscale | missing families | low | low to medium | measured: 136, 712, 194 and 584 fail |
| 7 | grouped GMM | missing; needs a loader change, because gfx1201 has no Gluon tree | none (dense) | medium for MoE that fits 32 GB | measured: 59 fail |
| 8 | MXFP8 `gemm_afp8wfp8*` | missing | none | low: no scaled WMMA, so `dot_scaled` is emulated | measured: 320 fail |
| — | FP4 families | out of scope | — | — | `is_any_fp4_avail()` is false |

### Time estimates

Card time is modelled, not measured:

- Compile: 2 to 6 s per config, serial per worker.
- Overhead: 0.2 s per config.
- Kernel time: a roofline at 640 GB/s, with BF16 at about 190 TFLOPS and FP8 at about
  380 TFLOPS. "lo" is the roofline; "hi" is 3× the roofline.
- The repo's own ETA is 5 s per config.
- There are 7 unique Qwen shapes (o_proj and GDN out_proj share one), with 13 M buckets
  each. lm_head uses decode M only.

**Tier A: seed a `DEFAULT.json`.**

- Copy the nearest seed: gfx1100 or gfx1151, both wave32 WMMA archs with 64 KB LDS.
  Otherwise use gfx942 and drop `kpack`.
- Check every bucket against 64 KiB LDS and 256 accumulators per thread at wave32.
- Run the family's op test.

This costs about 1 h of person-time per family and needs no sweep. There are 18
realistic families, about 2 to 3 working days in all. It turns crashes into working
kernels with `is_tuned = False`, which vLLM's gate still declines.

**Tier B: sweep one representative shape over the 13 M buckets** and copy the winners
into `DEFAULT.json`.

| family | configs | card time |
|---|---:|---|
| `gemm_a16w16` (14336, 5120) | 20,060 | 18 to 51 h |
| `gemm_a8w8_blockscale` | 5,968 | 4.5 to 13 h |

**Tier C: per-shape sweep** (`is_tuned = True`).

| family / shape set | configs | repo ETA | modelled |
|---|---:|---:|---|
| `gemm_a8w8_blockscale`, 6 linear shapes, M in {1, 4, 8, 16, 32, 64, 128}: the minimum for vLLM to take the aiter path at decode | 10,440 | 15 h | **6.6 to 18.7 h** |
| `gemm_a8w8_blockscale`, 7 shapes, reduced RDNA4 table | 37,160 | 52 h | 29 to 82 h |
| `gemm_a16w16`, 7 shapes, reduced table | 126,814 | 176 h | 116 to 335 h (5 to 14 days) |
| `gemm_a16w4`, 7 shapes, reduced table | 126,814 | 176 h | 113 to 324 h |
| `gemm_a16w16`, 7 shapes, full repo `SEARCH_SPACE` | 3,408,420 | 4,734 h | 3,198 to 9,215 h (133 to 384 days) |
| `gemm_a8w8_blockscale`, 7 shapes, full space | 505,570 | 702 h | 391 to 1,116 h |

The "reduced table" is an illustration, not proven to lose nothing. It uses:

- `BLOCK_SIZE_M/N` 16 to 256, `BLOCK_SIZE_K` 32 to 512
- `GROUP_SIZE_M` {1, 8}, `num_warps` {2, 4, 8}, `waves_per_eu` {1, 2}
- `cache_modifier` pinned
- `NUM_KSPLIT` {1, 3, 4, 8, 16}
- the wave32 register check

## 5. Phase 2 order

Each step is a separate change.

0. **Get a working timer.** `run_perftest` reads 0 µs in this image (section 1).
   Find out why the profiler records no kernel events, or time with HIP events or
   `triton.testing.do_bench`. Until then no port can be compared with its fallback.
1. **Triage the rope mismatches** (2,091 cases, about 1 bf16 ULP). Decide whether the
   kernel or the reference is wrong before anyone touches a tolerance.
2. **Tier A seeds** for `gemm_a16w16` and its six BF16 siblings, then `gemm_a16w4`.
   Each seed lands as its own commit (`configs/AGENTS.md` rule 5). Benchmark each
   against `torch.matmul` at the Qwen shapes: a seed slower than torch is reported,
   not merged.
3. **The tuner wave-size fix** (`tuning/space.py:219`), before any sweep.
4. **`module_top_k_per_row`**: fit the 12-bit radix instance into 64 KiB.
5. **FlyDSL GDR decode on wave32.** This is the highest-value port for Qwen3.8, but
   its production value also needs a vLLM change, because vLLM does not call it on
   RDNA4. It must beat the GDN decode path vLLM uses today. It needs a
   reference-vs-output test at every lane-count change.
6. **The one-line gates as one change:**
   - `opus.hpp` packed atomics
   - the `rope_common.h` OCP FP8 pack
   - the `qk_norm_rope_quant.py` w32 builder
7. **Add an OCP-FP8 MoE GEMM case for RDNA4** to `test_moe_gemm_a8w8.py`. Then the
   LDS cap fix, and any later MoE staging change, is tested on gfx1201.
8. **FlyDSL `causal_conv1d_update`**: add the arch and validate.
9. **Sweeps**, only where a consumer will take the result:
   - the minimal blockscale sweep (6.6 to 18.7 h) if an FP8-blockscale checkpoint is
     in use;
   - otherwise a tier-B `gemm_a16w16` default.

Open findings outside aiter's kernels, worth reporting but not fixing here:

- host memory growth in `test_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py` (about 40 GiB resident at a third of its cases)
- the `F.conv1d` abort in the torch reference
- the `rotate_activation_hd128` abort
- the out-of-memory test shapes, which need more than 32 GB
- the two `test_mha_with_sink` backward mismatches
