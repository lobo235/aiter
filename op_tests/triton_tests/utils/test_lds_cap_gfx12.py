# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The Triton LDS-cap table must know RDNA4, or pick_gemm_num_stages skips its check."""

import pytest

from aiter.jit.utils.lds_capacity import get_lds_capacity_bytes
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.gemm_config_utils import pick_gemm_num_stages


@pytest.mark.parametrize("arch", ["gfx1200", "gfx1201"])
def test_lds_cap_matches_jit_table(arch):
    # Two tables describe the same hardware limit; they must agree.
    assert arch_info._LDS_CAP_BYTES[arch] == get_lds_capacity_bytes(arch) == 64 * 1024


@pytest.mark.parametrize(
    "block_m, block_n, block_k, bits, use_async_padding, expected",
    [
        # BF16 non-async, 2 stages keep one extra A+B pair: 64 KiB fits exactly.
        (256, 256, 64, 16, False, 2),
        # BF16 non-async, the extra pair is 128 KiB: over the cap, drop to 1 stage.
        (256, 256, 128, 16, False, 1),
        # FP8 async-padded (the moe_op_gemm_a8w8 call): 2 stages x (A+B) fits.
        (64, 128, 128, 8, True, 2),
        # FP8 async-padded 128^3: ~72 KiB for 2 stages, over a 64 KiB cap.
        (128, 128, 128, 8, True, 1),
        # FP8 async-padded with 2 x (64 + 64 KiB) of tiles plus padding: far over the cap.
        (256, 256, 256, 8, True, 1),
    ],
)
@pytest.mark.parametrize("arch", ["gfx1200", "gfx1201"])
def test_pick_gemm_num_stages_gfx12(
    arch, block_m, block_n, block_k, bits, use_async_padding, expected
):
    assert (
        pick_gemm_num_stages(
            arch, block_m, block_n, block_k, bits, bits, use_async_padding
        )
        == expected
    )
