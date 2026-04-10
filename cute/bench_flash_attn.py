"""
Benchmark: flash_attn_simple block-sparse FA2 vs. torch SDPA vs. naive

Usage:
    python cute/bench_flash_attn.py
    python cute/bench_flash_attn.py --sparsity 0.0 0.5 0.75 0.9
    python cute/bench_flash_attn.py --dtype BFloat16 --seqlen_q 512 --seqlen_k 512

Sparsity = fraction of (m_block, n_block) pairs that are SKIPPED (set to 0 in mask).
0.0 = dense (all blocks active), 0.9 = 90% of blocks skipped.
"""

import argparse
import math

import torch
import torch.nn.functional as F

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack

from flash_attn_simple import FlashAttentionForwardAmpere


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def create_torch_tensors(B, Sq, Sk, H, D, torch_dtype):
    """Return Q, K, V in (B, H, S, D) layout."""
    q = torch.randn(B, H, Sq, D, dtype=torch_dtype, device="cuda")
    k = torch.randn(B, H, Sk, D, dtype=torch_dtype, device="cuda")
    v = torch.randn(B, H, Sk, D, dtype=torch_dtype, device="cuda")
    return q, k, v


def to_cute(t_bhsd: torch.Tensor, dtype) -> tuple:
    """(B,H,S,D) → (B,S,H,D) cute.Tensor + the underlying (B,S,H,D) torch tensor."""
    t = t_bhsd.permute(0, 2, 1, 3).contiguous()
    ct = (
        from_dlpack(t, assumed_align=16)
        .mark_layout_dynamic(leading_dim=3)
        .mark_compact_shape_dynamic(
            mode=3,
            stride_order=t.dim_order(),
            divisibility=(128 // dtype.width),
        )
    )
    return ct, t


def make_block_mask(B, H, Mb, Nb, sparsity: float, seed: int = 0) -> tuple:
    """
    Build a block mask at the requested sparsity level.
    sparsity=0.0 → all blocks active (dense).
    sparsity=0.9 → 90% of blocks skipped.
    Returns (torch int8 tensor on CUDA, cute.Tensor).
    """
    torch.manual_seed(seed)
    mask_t = (torch.rand(B, H, Mb, Nb) >= sparsity).to(torch.int8).cuda()
    # ensure at least one active block per query row (avoid all-zero output rows)
    for b in range(B):
        for h in range(H):
            for mi in range(Mb):
                if mask_t[b, h, mi].sum() == 0:
                    mask_t[b, h, mi, torch.randint(Nb, (1,))] = 1
    mask_c = from_dlpack(mask_t, assumed_align=1)
    return mask_t, mask_c


def make_o_cute(o_bshd: torch.Tensor, dtype) -> cute.Tensor:
    return (
        from_dlpack(o_bshd, assumed_align=16)
        .mark_layout_dynamic(leading_dim=3)
        .mark_compact_shape_dynamic(
            mode=3,
            stride_order=o_bshd.dim_order(),
            divisibility=(128 // dtype.width),
        )
    )


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def benchmark_naive(q, k, v, scale, warmup, iters):
    """Pure-PyTorch O(N²) attention."""
    def run():
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        return torch.matmul(torch.softmax(scores, dim=-1), v)

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / iters  # µs


def benchmark_sdpa(q, k, v, scale, warmup, iters):
    """torch.nn.functional.scaled_dot_product_attention."""
    torch.backends.cuda.enable_flash_sdp(True)

    for _ in range(warmup):
        F.scaled_dot_product_attention(q, k, v, scale=scale)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        F.scaled_dot_product_attention(q, k, v, scale=scale)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / iters  # µs


def benchmark_fa2(compiled_fa2, q_c, k_c, v_c, o_bshd, mask_c,
                  dtype, scale, cu_stream, warmup, iters):
    """Block-sparse FA2 kernel."""
    o_c = make_o_cute(o_bshd, dtype)

    for _ in range(warmup):
        compiled_fa2(q_c, k_c, v_c, o_c, mask_c, scale, cu_stream)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    torch.cuda.current_stream().record_event(start)
    for _ in range(iters):
        compiled_fa2(q_c, k_c, v_c, o_c, mask_c, scale, cu_stream)
    torch.cuda.current_stream().record_event(end)
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / iters  # µs


# ---------------------------------------------------------------------------
# FLOPs helper
# ---------------------------------------------------------------------------

def active_flops(B, Sq, Sk, H, D, active_fraction):
    """FLOPs on the active (non-masked) portion of attention."""
    # Q·Kᵀ + P·V, each B·H·Sq·Sk·D·2 MACs, scaled by fraction of active blocks
    return 2 * 2 * B * H * Sq * Sk * D * active_fraction


def fmt_row(label, time_us, tflops):
    return f"  {label:<30s}  {time_us:>9.1f} µs   {tflops:>7.2f} TFLOP/s"


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    dtype_str, batch_size, seqlen_q, seqlen_k, num_head, head_dim,
    softmax_scale, m_block_size, n_block_size, num_threads,
    sparsity_levels, warmup, iters, verify,
):
    dtype      = cutlass.dtype(dtype_str)
    torch_dtype = cutlass_torch.dtype(dtype)

    if not FlashAttentionForwardAmpere.can_implement(
        dtype, head_dim, m_block_size, n_block_size, num_threads
    ):
        print(f"[SKIP] {dtype_str} D={head_dim} m={m_block_size} n={n_block_size} T={num_threads}")
        return

    B, Sq, Sk, H, D = batch_size, seqlen_q, seqlen_k, num_head, head_dim
    Mb = math.ceil(Sq / m_block_size)
    Nb = math.ceil(Sk / n_block_size)

    print(f"\n{'='*68}")
    print(f"  dtype={dtype_str}  B={B}  Sq={Sq}  Sk={Sk}  H={H}  D={D}"
          f"  scale={softmax_scale:.3f}")
    print(f"  m_block={m_block_size}  n_block={n_block_size}  threads={num_threads}"
          f"  ({Mb}×{Nb} block grid)")
    print(f"{'='*68}")

    q_bhsd, k_bhsd, v_bhsd = create_torch_tensors(B, Sq, Sk, H, D, torch_dtype)
    q_c, q_bshd = to_cute(q_bhsd, dtype)
    k_c, k_bshd = to_cute(k_bhsd, dtype)
    v_c, v_bshd = to_cute(v_bhsd, dtype)
    o_bshd = torch.empty_like(q_bshd)

    torch_stream = torch.cuda.current_stream()
    cu_stream    = cuda.CUstream(torch_stream.cuda_stream)

    # Compile once using a dense mask as the template
    _, dense_mask_c = make_block_mask(B, H, Mb, Nb, sparsity=0.0)
    o_c_template   = make_o_cute(o_bshd, dtype)
    fa2            = FlashAttentionForwardAmpere(D, m_block_size, n_block_size, num_threads)
    compiled_fa2   = cute.compile(
        fa2, q_c, k_c, v_c, o_c_template, dense_mask_c, softmax_scale, cu_stream
    )

    # ---- baselines (dense, no block mask) -----------------------------------
    t_naive = benchmark_naive(q_bhsd, k_bhsd, v_bhsd, softmax_scale, warmup, iters)
    t_sdpa  = benchmark_sdpa(q_bhsd, k_bhsd, v_bhsd, softmax_scale, warmup, iters)

    dense_flops = active_flops(B, Sq, Sk, H, D, 1.0)
    print(fmt_row("naive attention",   t_naive, dense_flops / (t_naive * 1e-6) / 1e12))
    print(fmt_row("torch SDPA (dense)", t_sdpa, dense_flops / (t_sdpa  * 1e-6) / 1e12))
    print()

    # ---- FA2 at each sparsity level -----------------------------------------
    t_fa2_dense = None
    for sparsity in sparsity_levels:
        mask_t, mask_c = make_block_mask(B, H, Mb, Nb, sparsity)
        active_frac    = mask_t.float().mean().item()
        n_active       = int(mask_t.sum().item())
        n_total        = B * H * Mb * Nb

        # optional correctness check vs PyTorch block-sparse reference
        if verify and sparsity == 0.0:
            compiled_fa2(q_c, k_c, v_c, make_o_cute(o_bshd, dtype), mask_c,
                         softmax_scale, cu_stream)
            torch.cuda.synchronize()
            ref = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, scale=softmax_scale
            ).permute(0, 2, 1, 3)
            try:
                torch.testing.assert_close(o_bshd.cpu(), ref.cpu(), atol=1e-2, rtol=1e-4)
                print("  [verify dense]   PASS")
            except AssertionError as e:
                print(f"  [verify dense]   FAIL — {e}")

        t = benchmark_fa2(compiled_fa2, q_c, k_c, v_c, o_bshd, mask_c,
                          dtype, softmax_scale, cu_stream, warmup, iters)

        if sparsity == 0.0:
            t_fa2_dense = t

        af     = active_flops(B, Sq, Sk, H, D, active_frac)
        tflops = af / (t * 1e-6) / 1e12
        speedup_vs_dense = (t_fa2_dense / t) if t_fa2_dense else 1.0
        label  = f"FA2 sparse={sparsity:.0%} ({n_active}/{n_total})"
        print(fmt_row(label, t, tflops) +
              f"   {speedup_vs_dense:>5.2f}x vs dense FA2")

    print()
    if t_fa2_dense:
        print(fmt_row("FA2 dense vs SDPA speedup", t_sdpa, 0) +
              f"   {t_sdpa / t_fa2_dense:>5.2f}x")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_CONFIGS = [
    # (seqlen_q, seqlen_k, num_head, head_dim)
    # (196, 196, 200, 64),
    (196, 196, 200, 64),
]


def main():
    parser = argparse.ArgumentParser(description="Block-sparse FA2 benchmark")
    parser.add_argument("--dtype",         default="BFloat16")
    parser.add_argument("--batch_size",    type=int,   default=1)
    parser.add_argument("--seqlen_q",      type=int,   default=None)
    parser.add_argument("--seqlen_k",      type=int,   default=None)
    parser.add_argument("--num_head",      type=int,   default=None)
    parser.add_argument("--head_dim",      type=int,   default=None)
    parser.add_argument("--softmax_scale", type=float, default=None)
    parser.add_argument("--m_block_size",  type=int,   default=32)
    parser.add_argument("--n_block_size",  type=int,   default=32)
    parser.add_argument("--num_threads",   type=int,   default=64)
    parser.add_argument("--sparsity",      type=float, nargs="+",
                        default=[0.0, 0.25, 0.5, 0.75, 0.9, 0.95],
                        help="List of sparsity levels to sweep (0=dense, 0.9=90%% blocks skipped)")
    parser.add_argument("--warmup",        type=int,   default=5)
    parser.add_argument("--iters",         type=int,   default=20)
    parser.add_argument("--no_verify",     action="store_true")
    args = parser.parse_args()

    if args.seqlen_q is not None:
        configs = [(args.seqlen_q, args.seqlen_k or args.seqlen_q,
                    args.num_head or 16, args.head_dim or 64)]
    else:
        configs = DEFAULT_CONFIGS

    for sq, sk, nh, hd in configs:
        scale = args.softmax_scale or (1.0 / math.sqrt(hd))
        run_benchmark(
            dtype_str     = args.dtype,
            batch_size    = args.batch_size,
            seqlen_q      = sq,
            seqlen_k      = sk,
            num_head      = nh,
            head_dim      = hd,
            softmax_scale = scale,
            m_block_size  = args.m_block_size,
            n_block_size  = args.n_block_size,
            num_threads   = args.num_threads,
            sparsity_levels = sorted(set(args.sparsity)),
            warmup        = args.warmup,
            iters         = args.iters,
            verify        = not args.no_verify,
        )


if __name__ == "__main__":
    main()
