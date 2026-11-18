# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""run_perftest must report a real time, even where the torch profiler records no GPU kernels."""

import torch

from aiter.test_common import run_perftest


def test_run_perftest_reports_nonzero_time():
    a = torch.randn(2048, 2048, device="cuda")
    _, us = run_perftest(torch.matmul, a, a, num_iters=20, num_warmup=2)
    assert us > 0


def test_run_perftest_graph_reports_nonzero_time():
    a = torch.randn(2048, 2048, device="cuda")
    _, us = run_perftest(torch.matmul, a, a, num_iters=20, num_warmup=2, testGraph=True)
    assert us > 0


if __name__ == "__main__":
    test_run_perftest_reports_nonzero_time()
    test_run_perftest_graph_reports_nonzero_time()
    print("ok")
