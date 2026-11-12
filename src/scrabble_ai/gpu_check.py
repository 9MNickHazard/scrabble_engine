"""Verify that PyTorch does real work on the GPU, not just that it reports one.

`torch.cuda.is_available()` can be true for a build whose kernels do not support the
card's architecture, so this runs a matmul and a small bf16 training step on the device
and compares the matmul with the CPU result.
"""

from __future__ import annotations

import time


def run() -> int:
    import torch

    print(f"torch            {torch.__version__}")
    print(f"built for CUDA   {torch.version.cuda}")
    print(f"cuDNN            {torch.backends.cudnn.version()}")
    print(f"cuda available   {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("FAIL: no CUDA device visible to this PyTorch build")
        return 1

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    major, minor = torch.cuda.get_device_capability(device)
    arch = f"sm_{major}{minor}"
    arch_list = torch.cuda.get_arch_list()
    print(f"device           {props.name}")
    print(f"memory           {props.total_memory / 2**30:.1f} GiB")
    print(f"capability       {arch}")
    print(f"build targets    {' '.join(arch_list)}")
    ok = True
    if arch not in arch_list:
        print(f"FAIL: this build has no kernels for {arch}")
        ok = False

    # 1. Matmul on the GPU, checked against the CPU.
    generator = torch.Generator().manual_seed(0)
    a = torch.randn(4096, 4096, generator=generator)
    b = torch.randn(4096, 4096, generator=generator)
    start = time.perf_counter()
    cpu_result = a @ b
    cpu_seconds = time.perf_counter() - start

    a_gpu, b_gpu = a.to(device), b.to(device)
    (a_gpu @ b_gpu).sum().item()  # warm-up: context creation, cuBLAS load
    torch.cuda.synchronize()
    start = time.perf_counter()
    gpu_result = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_seconds = time.perf_counter() - start

    relative_error = ((gpu_result.cpu() - cpu_result).abs().max() / cpu_result.abs().max()).item()
    print(f"matmul 4096x4096 CPU {cpu_seconds * 1e3:.1f} ms | GPU {gpu_seconds * 1e3:.2f} ms | "
          f"max relative difference {relative_error:.2e}")
    if not (gpu_result.device.type == "cuda" and relative_error < 1e-3):
        print("FAIL: GPU matmul disagrees with the CPU")
        ok = False

    # 2. One bf16-autocast training step on a small conv net: the path training will use.
    net = torch.nn.Sequential(
        torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(),
        torch.nn.Conv2d(64, 64, 3, padding=1), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(64, 1),
    ).to(device).to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    boards = torch.randn(256, 32, 15, 15, device=device).to(memory_format=torch.channels_last)
    targets = torch.randn(256, 1, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = torch.nn.functional.mse_loss(net(boards).float(), targets)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    grads_finite = all(torch.isfinite(p.grad).all().item() for p in net.parameters())
    print(f"bf16 conv train step: loss {loss.item():.4f}, gradients finite: {grads_finite}")
    if not (torch.isfinite(loss).item() and grads_finite):
        print("FAIL: bf16 training step produced non-finite values")
        ok = False

    print("RESULT: " + ("OK - PyTorch is computing on the GPU" if ok else "FAILED"))
    return 0 if ok else 1
