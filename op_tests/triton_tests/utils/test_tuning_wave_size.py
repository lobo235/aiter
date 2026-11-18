# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The tuner's register filter must use the arch's real wave size (32 on RDNA, 64 on CDNA)."""

import os
import sys
from types import SimpleNamespace

import pytest

# The tuner is a script directory, not a package: its modules import each other by bare name.
from aiter.ops.triton.utils._triton import arch_info

sys.path.insert(0, os.path.join(os.path.dirname(arch_info.__file__), "tuning"))
from space import exceeds_registers

SPEC = SimpleNamespace(cta_split=None)
# 256x256 fp32 accumulator over 4 warps: 256 values per lane at wave64, 512 at wave32.
CONFIG = {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "num_warps": 4}


@pytest.mark.parametrize("arch", ["gfx1100", "gfx1151", "gfx1200", "gfx1201"])
def test_rdna_counts_32_lanes(arch):
    assert exceeds_registers(CONFIG, SPEC, arch)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_cdna_counts_64_lanes(arch):
    assert not exceeds_registers(CONFIG, SPEC, arch)


def test_rdna_tile_that_fits():
    # 256x256 over 8 warps at wave32 is exactly 256 values per lane: allowed.
    assert not exceeds_registers({**CONFIG, "num_warps": 8}, SPEC, "gfx1201")
