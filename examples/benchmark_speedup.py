"""
Benchmark and plot speedup for block-sparse attention vs dense baseline.

Runs a dense baseline (matmul softmax) and block-sparse kernel across a
set of sparsity levels, then saves a speedup plot.
"""

import argparse
import os
import sys
import time
from typing import List, Tuple

import torch
import torch.nn.functional as F

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from block_sparse_attn.attention import block_sparse_attn_simple


def diagonal_band(n: int, k: int, device: str) -> torch.Tensor:
    idx = torch.arange(n, device=device)
    dist = (idx[:, None] - idx[None, :]).abs()
    return (dist <= k)


def make_band_mask(
    batch_size: int,
    num_heads: int,
    nrow: int,
    ncol: int,
    keep_ratio: float,
    device: str,
) -> torch.Tensor:
    band = diagonal_band(nrow, k=max(1, int(keep_ratio * nrow)), device=device)
    band = band[:nrow, :ncol]
    return band[None, None, ...].expand(batch_size, num_heads, nrow, ncol)

def estimate_dense_attention_flops(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
) -> float:
    # QK^T and AV matmuls: each costs ~2 * B * H * L * L * D FLOPs
    return 4.0 * batch_size * num_heads * seq_len * seq_len * head_dim


def time_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positional: torch.Tensor,
    iters: int,
    warmup: int,
    head_dim: int,
) -> float:
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)

    for _ in range(warmup):
        attn = torch.matmul(q_t * (head_dim ** -0.5), k_t.transpose(-2, -1))
        attn = attn + positional
        attn = torch.softmax(attn, dim=-1)
        _ = torch.matmul(attn, v_t).transpose(1, 2)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        attn = torch.matmul(q_t * (head_dim ** -0.5), k_t.transpose(-2, -1))
        attn = attn + positional
        attn = torch.softmax(attn, dim=-1)
        _ = torch.matmul(attn, v_t).transpose(1, 2)
    torch.cuda.synchronize()
    return (time.time() - start) / iters


def time_sparse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    positional: torch.Tensor,
    iters: int,
    warmup: int,
) -> float:
    for _ in range(warmup):
        _ = block_sparse_attn_simple(q, k, v, mask, positional=positional)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        _ = block_sparse_attn_simple(q, k, v, mask, positional=positional)
    torch.cuda.synchronize()
    return (time.time() - start) / iters

def get_decomposed_rel_pos(
        self,
        q: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size,
        k_size, 
    ):
        q_h, q_w = q_size
        k_h, k_w = k_size
        Rh = get_rel_pos(q_h, k_h, rel_pos_h)
        Rw = get_rel_pos(q_w, k_w, rel_pos_w)
        
        B = q.shape[0]
        r_q = q.view(B, q_h, q_w, -1)
        
        # Optimized einsum
        rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
        rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
        
        # Fused broadcasting and reshaping
        return (rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).flatten(1, 2).flatten(2, 3)

def benchmark(
    seq_len: int,
    batch_size: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    keep_ratios: List[float],
    iters: int,
    warmup: int,
    dtype: torch.dtype,
) -> Tuple[List[float], List[float], float, float, float]:
    device = "cuda"
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    positional = torch.rand(batch_size, num_heads, seq_len, seq_len, device=device, dtype=dtype)
    

    nrow = (seq_len + block_size - 1) // block_size
    ncol = nrow

    dense_time = time_dense(q, k, v, positional, iters, warmup, head_dim)
    dense_mask = torch.ones(batch_size, num_heads, nrow, ncol, device=device, dtype=torch.bool)
    dense_kernel_time = time_sparse(q, k, v, dense_mask, positional, iters, warmup)
    dense_flops = estimate_dense_attention_flops(batch_size, num_heads, seq_len, head_dim)

    sparse_times = []
    eff_sparsities = []
    for keep_ratio in keep_ratios:
        mask = make_band_mask(batch_size, num_heads, nrow, ncol, keep_ratio, device)
        eff_sparsity = 1.0 - mask.float().mean().item()
        sparse_time = time_sparse(q, k, v, mask, positional, iters, warmup)
        eff_sparsities.append(eff_sparsity)
        sparse_times.append(sparse_time)

    return eff_sparsities, sparse_times, dense_time, dense_kernel_time, dense_flops


def plot_speedup(
    sparsities_by_label: List[Tuple[str, List[float]]],
    speedups_by_label: List[Tuple[str, List[float]]],
    dense_kernel_speedups_by_label: List[Tuple[str, float]],
    out_path: str,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    # Palette approximated from the provided figure.
    palette = [
        "#433b69",  # deep indigo
        "#3f6f8e",  # steel blue
        "#53b3a1",  # teal
        "#417505",  # green (fallback)
    ]
    for idx, ((label, sparsities), (_, speedups)) in enumerate(
        zip(sparsities_by_label, speedups_by_label)
    ):
        color = palette[idx % len(palette)]
        plt.plot([s * 100 for s in sparsities], speedups, marker="o", label=label, color=color)
    # for idx, (label, dense_kernel_speedup) in enumerate(dense_kernel_speedups_by_label):
    #     color = palette[idx % len(palette)]
    #     plt.axhline(
    #         dense_kernel_speedup,
    #         color=color,
    #         linestyle="-.",
    #         linewidth=1.2,
    #         label=f"{label} (dense)",
    #         alpha=0.9,
    #     )
    # Baseline: dense speedup = 1.0
    plt.axhline(1.0, color="grey", linestyle="--", linewidth=2.0, label="Baseline (1.0x)")
    plt.xlabel("Sparsity (%)")
    plt.ylabel("Speedup vs Dense")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    if len(speedups_by_label) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(f'{out_path}.jpg', dpi=300, format="jpg",  bbox_inches="tight"   )
    print(f"Saved plot to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark block-sparse speedup.")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batch-sizes", type=str, default="")
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--keep-ratios", type=str, default="1.0,0.75,0.5,0.25, 0.1, 0.05")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--out", type=str, default="benchmark_speedup.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; please run on a GPU.")
        return

    dtype = getattr(torch, args.dtype)
    keep_ratios = [float(x) for x in args.keep_ratios.split(",") if x.strip()]

    batch_sizes = [args.batch_size]
    if args.batch_sizes:
        batch_sizes = [int(x)*16 for x in args.batch_sizes.split(",") if x.strip()]

    print(batch_sizes)
    sparsities_by_label = []
    speedups_by_label = []
    dense_kernel_speedups_by_label = []

    for bsz in batch_sizes:
        sparsities, sparse_times, dense_time, dense_kernel_time, dense_flops = benchmark(
            seq_len=args.seq_len,
            batch_size=bsz,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            block_size=args.block_size,
            keep_ratios=keep_ratios,
            iters=args.iters,
            warmup=args.warmup,
            dtype=dtype,
        )
        label = f"Batch size B={bsz//16}"
        sparsities_by_label.append((label, sparsities))
        speedups_by_label.append((label, [dense_time / t for t in sparse_times]))
        dense_kernel_speedups_by_label.append((label, dense_time / dense_kernel_time))

        print(f"\nResults (batch size {bsz}):")
        for s, t in zip(sparsities, sparse_times):
            density = 1.0 - s
            sparse_flops = dense_flops * density
            print(
                f"  sparsity={s:.2%}  time={t*1000:.2f} ms  "
                f"speedup={dense_time/t:.2f}x  "
                f"est_gflops={sparse_flops/1e9:.2f}"
            )
        print(f"  dense_est_gflops={dense_flops/1e9:.2f}")

    title = (
        f"SparseSam Global Attention Speedup"
    )
    plot_speedup(
        sparsities_by_label,
        speedups_by_label,
        dense_kernel_speedups_by_label,
        args.out,
        title,
    )


if __name__ == "__main__":
    main()
