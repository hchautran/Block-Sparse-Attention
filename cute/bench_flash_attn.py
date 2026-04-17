"""
Benchmark: SAM attention (naive) vs. FA2 with decomposed rel_pos bias.

Usage:
    python cute/bench_flash_attn.py
    python cute/bench_flash_attn.py --model sam_h --batch_size 25
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

from flash_attn import FlashAttentionForwardAmpere


# ---------------------------------------------------------------------------
# SAM rel_pos helpers
# ---------------------------------------------------------------------------

def get_rel_pos(q_size, k_size, rel_pos):
    """rel_pos param (2*win-1, D) → (q_size, k_size, D) matrix."""
    max_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_dist:
        rel_pos = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_dist, mode="linear",
        ).reshape(-1, max_dist).permute(1, 0)
    q_coords = torch.arange(q_size, device=rel_pos.device)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size, device=rel_pos.device)[None, :] * max(q_size / k_size, 1.0)
    return rel_pos[(q_coords - k_coords + (k_size - 1) * max(q_size / k_size, 1.0)).long()]


def compute_rel_bias(q_bshd, Rh, Rw, win):
    """SAM einsum: (B,Sq,H,D) + Rh/Rw → rel_h, rel_w both (B,Sq,H,win) for FA2."""
    B, Sq, H, D = q_bshd.shape
    r_q   = q_bshd.permute(0, 2, 1, 3).reshape(B * H, win, win, D).float()
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh).reshape(B * H, Sq, win)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw).reshape(B * H, Sq, win)
    to_fa2 = lambda t: t.reshape(B, H, Sq, win).permute(0, 2, 1, 3).contiguous()
    return to_fa2(rel_h), to_fa2(rel_w)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_qkvo(B, S, H, D, torch_dtype, dtype):
    t = torch.randn(B, S, H, D, dtype=torch_dtype, device="cuda")
    ct = (from_dlpack(t, assumed_align=16)
          .mark_layout_dynamic(leading_dim=3)
          .mark_compact_shape_dynamic(mode=3, stride_order=t.dim_order(),
                                      divisibility=128 // dtype.width))
    return ct, t


def to_cute_pos(t):
    return from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=3)


def make_o_cute(o_t, dtype):
    return (from_dlpack(o_t, assumed_align=16)
            .mark_layout_dynamic(leading_dim=3)
            .mark_compact_shape_dynamic(mode=3, stride_order=o_t.dim_order(),
                                        divisibility=128 // dtype.width))


def timeit(fn, warmup, iters):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters   # µs


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(dtype_str, B, Sq, H, D, scale, m_block, n_block, threads,
                  warmup, iters, verify):
    dtype       = cutlass.dtype(dtype_str)
    torch_dtype = cutlass_torch.dtype(dtype)

    if not FlashAttentionForwardAmpere.can_implement(dtype, D, m_block, n_block, threads):
        print(f"[SKIP] {dtype_str} D={D} m={m_block} n={n_block} T={threads}")
        return

    win = int(math.sqrt(Sq))
    assert win * win == Sq, f"Sq={Sq} must be a perfect square"

    print(f"\n{'='*66}")
    print(f"  {dtype_str}  B={B}  Sq={Sq} (win={win}²)  H={H}  D={D}  scale={scale:.4f}")
    print(f"  m={m_block}  n={n_block}  threads={threads}")
    print(f"{'='*66}")

    q_c, q_t = make_qkvo(B, Sq, H, D, torch_dtype, dtype)
    k_c, k_t = make_qkvo(B, Sq, H, D, torch_dtype, dtype)
    v_c, v_t = make_qkvo(B, Sq, H, D, torch_dtype, dtype)
    o_c, o_t = make_qkvo(B, Sq, H, D, torch_dtype, dtype)
    print(q_t.shape)

    # SAM-style learnable rel_pos params (2*win-1, D), float32
    Rh = get_rel_pos(win, win, torch.randn(2*win-1, D, device="cuda"))
    Rw = get_rel_pos(win, win, torch.randn(2*win-1, D, device="cuda"))

    # Pre-computed biases (B, Sq, H, win) — static for this benchmark iteration
    rel_h_t, rel_w_t = compute_rel_bias(q_t, Rh, Rw, win)
    rel_h_c, rel_w_c = to_cute_pos(rel_h_t), to_cute_pos(rel_w_t)

    cu_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compiled = cute.compile(
        FlashAttentionForwardAmpere(D, m_block, n_block, threads, win),
        q_c, k_c, v_c, o_c, rel_h_c, rel_w_c, scale, cu_stream,
        options="",
    )

    # ---- verify ----
    if verify:
        compiled(q_c, k_c, v_c, o_c, rel_h_c, rel_w_c, scale, cu_stream)
        torch.cuda.synchronize()
        BH = B * H
        r_q = q_t.permute(0,2,1,3).reshape(BH, win, win, D).float()
        rh  = torch.einsum("bhwc,hkc->bhwk", r_q, Rh).reshape(BH, Sq, win)
        rw  = torch.einsum("bhwc,wkc->bhwk", r_q, Rw).reshape(BH, Sq, win)
        rp  = (rh[:,:,:,None] + rw[:,:,None,:]).reshape(BH, Sq, Sq)
        q_f = q_t.permute(0,2,1,3).reshape(BH, Sq, D).float()
        k_f = k_t.permute(0,2,1,3).reshape(BH, Sq, D).float()
        v_f = v_t.permute(0,2,1,3).reshape(BH, Sq, D).float()
        ref = (torch.matmul(F.softmax(q_f @ k_f.transpose(-2,-1) * scale + rp, dim=-1), v_f)
               .reshape(B, H, Sq, D).permute(0,2,1,3).to(torch_dtype))
        try:
            torch.testing.assert_close(o_t.cpu(), ref.cpu(), atol=1e-2, rtol=1e-4)
            print("  [verify] PASS")
        except AssertionError as e:
            print(f"  [verify] FAIL — {e}")

    # ---- SAM naive (einsum + matmul chain, float32) ----
    BH = B * H
    k_f = k_t.permute(0,2,1,3).reshape(BH, Sq, D).float()
    v_f = v_t.permute(0,2,1,3).reshape(BH, Sq, D).float()
    q_f = q_t.float()
    def sam_naive():
        r_q = q_f.permute(0,2,1,3).reshape(BH, win, win, D)
        rh = torch.einsum("bhwc,hkc->bhwk", r_q, Rh).reshape(BH, Sq, win)
        rw = torch.einsum("bhwc,wkc->bhwk", r_q, Rw).reshape(BH, Sq, win)
        rp = (rh[:,:,:,None] + rw[:,:,None,:]).reshape(BH, Sq, Sq)
        q  = q_f.permute(0,2,1,3).reshape(BH, Sq, D)
        return torch.matmul(F.softmax(q @ k_f.transpose(-2,-1) * scale + rp, dim=-1), v_f)

    # ---- FA2: einsum precompute + kernel ----
    def fa2_total():
        rh_t, rw_t = compute_rel_bias(q_t, Rh, Rw, win)
        print(rh_t.shape)
        compiled(q_c, k_c, v_c, make_o_cute(o_t, dtype),
                 to_cute_pos(rh_t), to_cute_pos(rw_t), scale, cu_stream)

    # ---- FA2: kernel only (bias pre-computed outside) ----
    def fa2_kernel():
        compiled(q_c, k_c, v_c, make_o_cute(o_t, dtype),
                 rel_h_c, rel_w_c, scale, cu_stream)

    t_sam  = timeit(sam_naive,  warmup, iters)
    t_fa2  = timeit(fa2_total,  warmup, iters)
    t_kern = timeit(fa2_kernel, warmup, iters)
    flops  = 4 * B * H * Sq * Sq * D   # 2×(QK + PV)

    def row(label, t):
        return f"  {label:<38s}  {t:>8.1f} µs   {flops/(t*1e-6)/1e12:>6.2f} TFLOP/s"

    print(row("SAM naive  (einsum + matmul)", t_sam))
    print(row("FA2 total  (einsum + kernel)", t_fa2))
    print(row("FA2 kernel (bias pre-cached)", t_kern))
    print()
    print(f"  FA2-total  vs SAM  : {t_sam/t_fa2:.2f}x")
    print(f"  FA2-kernel vs SAM  : {t_sam/t_kern:.2f}x")
    print(f"  einsum overhead    : ~{t_fa2-t_kern:.1f} µs")


# ---------------------------------------------------------------------------
# SAM presets  (effective batch = B × num_windows, win=14 → 25 windows)
# ---------------------------------------------------------------------------
SAM_CONFIGS = {
    "sam_l": (196, 16, 64),
    "sam_h": (196, 16, 80),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",         default=None, choices=list(SAM_CONFIGS))
    p.add_argument("--dtype",         default="BFloat16")
    p.add_argument("--batch_size",    type=int,   default=25)
    p.add_argument("--seqlen_q",      type=int,   default=None)
    p.add_argument("--num_head",      type=int,   default=None)
    p.add_argument("--head_dim",      type=int,   default=None)
    p.add_argument("--softmax_scale", type=float, default=None)
    p.add_argument("--m_block_size",  type=int,   default=32)
    p.add_argument("--n_block_size",  type=int,   default=32)
    p.add_argument("--num_threads",   type=int,   default=64)
    p.add_argument("--warmup",        type=int,   default=5)
    p.add_argument("--iters",         type=int,   default=20)
    p.add_argument("--no_verify",     action="store_true")
    args = p.parse_args()

    if args.model:
        configs = [SAM_CONFIGS[args.model]]
    elif args.seqlen_q:
        configs = [(args.seqlen_q, args.num_head or 16, args.head_dim or 64)]
    else:
        configs = [SAM_CONFIGS["sam_l"], SAM_CONFIGS["sam_h"]]

    for sq, nh, hd in configs:
        run_benchmark(
            dtype_str = args.dtype,
            B         = args.batch_size,
            Sq        = sq, H = nh, D = hd,
            scale     = args.softmax_scale or 1.0 / math.sqrt(hd),
            m_block   = args.m_block_size,
            n_block   = args.n_block_size,
            threads   = args.num_threads,
            warmup    = args.warmup,
            iters     = args.iters,
            verify    = not args.no_verify,
        )


if __name__ == "__main__":
    main()
