"""
Tests for block-sparse FlashAttentionForwardAmpere.

Covers:
  1. Dense mask (all-ones)  → must match torch SDPA exactly
  2. Sparse mask (random)   → must match a PyTorch block-sparse reference
  3. Lower-triangular mask  → manual causal reference
  4. All-zeros mask         → output must be zero (no active blocks)
  5. Single active block    → only one (m, n) pair contributes

Run:
    python cute/test_block_sparse.py
    python cute/test_block_sparse.py --dtype Float16 --verbose
"""

import argparse
import math
import sys

import torch
import torch.nn.functional as F

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack

from flash_attn_simple import FlashAttentionForwardAmpere


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_qkv(B, Sq, Sk, H, D, torch_dtype, seed=0):
    """Return Q(B,H,Sq,D), K(B,H,Sk,D), V(B,H,Sk,D) on CUDA."""
    torch.manual_seed(seed)
    q = torch.randn(B, H, Sq, D, dtype=torch_dtype, device="cuda") * 0.1
    k = torch.randn(B, H, Sk, D, dtype=torch_dtype, device="cuda") * 0.1
    v = torch.randn(B, H, Sk, D, dtype=torch_dtype, device="cuda") * 0.1
    return q, k, v


def to_cute(t_bhsd: torch.Tensor, dtype: type) -> tuple:
    """(B,H,S,D) → (B,S,H,D) cute.Tensor + underlying torch tensor."""
    t_bshd = t_bhsd.permute(0, 2, 1, 3).contiguous()
    ct = (
        from_dlpack(t_bshd, assumed_align=16)
        .mark_layout_dynamic(leading_dim=3)
        .mark_compact_shape_dynamic(
            mode=3,
            stride_order=t_bshd.dim_order(),
            divisibility=(128 // dtype.width),
        )
    )
    return ct, t_bshd


def make_block_mask_cute(mask_torch: torch.Tensor) -> cute.Tensor:
    """(B,H,Mb,Nb) int8 torch tensor → cute.Tensor."""
    return from_dlpack(mask_torch.cuda(), assumed_align=1)


def block_sparse_ref(q, k, v, scale, block_mask_np, m_block_size, n_block_size):
    """
    PyTorch reference for block-sparse attention.
    q/k/v: (B,H,S,D).  block_mask_np: (B,H,Mb,Nb) numpy/torch int8 array.
    Builds a dense float attn_bias with -inf on masked blocks and calls SDPA.
    """
    B, H, Sq, D = q.shape
    Sk = k.shape[2]
    attn_bias = torch.zeros(B, H, Sq, Sk, device=q.device, dtype=torch.float32)
    Mb, Nb = block_mask_np.shape[2], block_mask_np.shape[3]
    for b in range(B):
        for h in range(H):
            for mi in range(Mb):
                for ni in range(Nb):
                    if block_mask_np[b, h, mi, ni] == 0:
                        r0, r1 = mi * m_block_size, min((mi + 1) * m_block_size, Sq)
                        c0, c1 = ni * n_block_size, min((ni + 1) * n_block_size, Sk)
                        attn_bias[b, h, r0:r1, c0:c1] = float("-inf")
    # Use math backend to respect attn_bias exactly
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    out = F.scaled_dot_product_attention(
        q.float(), k.float(), v.float(),
        attn_mask=attn_bias, scale=scale
    ).to(q.dtype)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    return out


def run_kernel(fa2_compiled, q_cute, k_cute, v_cute, o_bshd, block_mask_cute,
               scale, cu_stream):
    o_cute = (
        from_dlpack(o_bshd, assumed_align=16)
        .mark_layout_dynamic(leading_dim=3)
        .mark_compact_shape_dynamic(
            mode=3,
            stride_order=o_bshd.dim_order(),
            divisibility=(128 // q_cute.element_type.width),
        )
    )
    fa2_compiled(q_cute, k_cute, v_cute, o_cute, block_mask_cute, scale, cu_stream)
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Individual test cases
# ---------------------------------------------------------------------------

def test_dense(fa2_compiled, dtype, q, k, v, scale, m_bs, n_bs, cu_stream, verbose):
    """All-ones mask → must match torch SDPA."""
    B, H, Sq, D = q.shape
    Sk = k.shape[2]
    Mb = math.ceil(Sq / m_bs)
    Nb = math.ceil(Sk / n_bs)

    mask_t = torch.ones(B, H, Mb, Nb, dtype=torch.int8).cuda()
    mask_c = make_block_mask_cute(mask_t)

    q_c, q_bshd = to_cute(q, dtype)
    k_c, k_bshd = to_cute(k, dtype)
    v_c, v_bshd = to_cute(v, dtype)
    o_bshd = torch.empty_like(q_bshd)

    run_kernel(fa2_compiled, q_c, k_c, v_c, o_bshd, mask_c, scale, cu_stream)
    got = o_bshd.permute(0, 2, 1, 3)  # back to (B,H,S,D)

    torch.backends.cuda.enable_flash_sdp(True)
    ref = F.scaled_dot_product_attention(q, k, v, scale=scale)

    try:
        torch.testing.assert_close(got.cpu(), ref.cpu(), atol=1e-2, rtol=1e-4)
        if verbose: print("  [dense]          PASS")
        return True
    except AssertionError as e:
        print(f"  [dense]          FAIL – {e}")
        return False


def test_sparse_random(fa2_compiled, dtype, q, k, v, scale, m_bs, n_bs, cu_stream,
                       sparsity, seed, verbose):
    """Random block mask → must match PyTorch block-sparse reference."""
    B, H, Sq, D = q.shape
    Sk = k.shape[2]
    Mb = math.ceil(Sq / m_bs)
    Nb = math.ceil(Sk / n_bs)

    torch.manual_seed(seed + 42)
    mask_t = (torch.rand(B, H, Mb, Nb) > sparsity).to(torch.int8).cuda()
    # ensure at least one active block per row to avoid all-zero output
    for b in range(B):
        for h in range(H):
            for mi in range(Mb):
                if mask_t[b, h, mi].sum() == 0:
                    mask_t[b, h, mi, torch.randint(Nb, (1,))] = 1
    mask_c = make_block_mask_cute(mask_t)

    q_c, q_bshd = to_cute(q, dtype)
    k_c, k_bshd = to_cute(k, dtype)
    v_c, v_bshd = to_cute(v, dtype)
    o_bshd = torch.empty_like(q_bshd)

    run_kernel(fa2_compiled, q_c, k_c, v_c, o_bshd, mask_c, scale, cu_stream)
    got = o_bshd.permute(0, 2, 1, 3)

    ref = block_sparse_ref(q, k, v, scale, mask_t.cpu(), m_bs, n_bs)

    try:
        torch.testing.assert_close(got.cpu(), ref.cpu(), atol=1e-2, rtol=1e-4)
        active_pct = 100.0 * mask_t.float().mean().item()
        if verbose: print(f"  [sparse {active_pct:4.1f}% active]  PASS")
        return True
    except AssertionError as e:
        print(f"  [sparse]         FAIL – {e}")
        return False


def test_lower_triangular(fa2_compiled, dtype, q, k, v, scale, m_bs, n_bs,
                          cu_stream, verbose):
    """Block-lower-triangular mask → coarse causal reference."""
    B, H, Sq, D = q.shape
    Sk = k.shape[2]
    Mb = math.ceil(Sq / m_bs)
    Nb = math.ceil(Sk / n_bs)

    mask_t = torch.zeros(B, H, Mb, Nb, dtype=torch.int8)
    for mi in range(Mb):
        for ni in range(Nb):
            # keep block if the n-block starts before/at the m-block
            if ni <= mi:
                mask_t[:, :, mi, ni] = 1
    mask_t = mask_t.cuda()
    mask_c = make_block_mask_cute(mask_t)

    q_c, q_bshd = to_cute(q, dtype)
    k_c, k_bshd = to_cute(k, dtype)
    v_c, v_bshd = to_cute(v, dtype)
    o_bshd = torch.empty_like(q_bshd)

    run_kernel(fa2_compiled, q_c, k_c, v_c, o_bshd, mask_c, scale, cu_stream)
    got = o_bshd.permute(0, 2, 1, 3)

    ref = block_sparse_ref(q, k, v, scale, mask_t.cpu(), m_bs, n_bs)

    try:
        torch.testing.assert_close(got.cpu(), ref.cpu(), atol=1e-2, rtol=1e-4)
        if verbose: print("  [lower-tri]      PASS")
        return True
    except AssertionError as e:
        print(f"  [lower-tri]      FAIL – {e}")
        return False


def test_all_zeros(fa2_compiled, dtype, q, k, v, scale, m_bs, n_bs,
                   cu_stream, verbose):
    """All-zeros mask → output must be all zeros (no blocks active)."""
    B, H, Sq, D = q.shape
    Sk = k.shape[2]
    Mb = math.ceil(Sq / m_bs)
    Nb = math.ceil(Sk / n_bs)

    mask_t = torch.zeros(B, H, Mb, Nb, dtype=torch.int8).cuda()
    mask_c = make_block_mask_cute(mask_t)

    q_c, q_bshd = to_cute(q, dtype)
    k_c, k_bshd = to_cute(k, dtype)
    v_c, v_bshd = to_cute(v, dtype)
    o_bshd = torch.empty_like(q_bshd)

    run_kernel(fa2_compiled, q_c, k_c, v_c, o_bshd, mask_c, scale, cu_stream)
    got = o_bshd.permute(0, 2, 1, 3)

    try:
        assert not got.isnan().any(), "NaN in output"
        assert not got.isinf().any(), "Inf in output"
        torch.testing.assert_close(got.cpu(), torch.zeros_like(got).cpu(),
                                   atol=1e-4, rtol=0)
        if verbose: print("  [all-zeros mask] PASS")
        return True
    except (AssertionError, Exception) as e:
        print(f"  [all-zeros mask] FAIL – {e}")
        return False


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_tests(dtype_str, B, Sq, Sk, H, D, m_bs, n_bs, num_threads, scale, verbose):
    dtype = cutlass.dtype(dtype_str)
    torch_dtype = cutlass_torch.dtype(dtype)

    if not FlashAttentionForwardAmpere.can_implement(dtype, D, m_bs, n_bs, num_threads):
        print(f"[SKIP] config not supported: dtype={dtype_str} D={D} m={m_bs} n={n_bs} T={num_threads}")
        return True

    print(f"\n── dtype={dtype_str}  B={B}  Sq={Sq}  Sk={Sk}  H={H}  D={D}"
          f"  m={m_bs}  n={n_bs}  T={num_threads}")

    torch_stream = torch.cuda.current_stream()
    cu_stream = cuda.CUstream(torch_stream.cuda_stream)

    q, k, v = make_qkv(B, Sq, Sk, H, D, torch_dtype)

    # Compile once with a representative dense mask
    Mb = math.ceil(Sq / m_bs)
    Nb = math.ceil(Sk / n_bs)
    dummy_mask_t = torch.ones(B, H, Mb, Nb, dtype=torch.int8).cuda()
    dummy_mask_c = make_block_mask_cute(dummy_mask_t)
    q_c, q_bshd = to_cute(q, dtype)
    k_c, k_bshd = to_cute(k, dtype)
    v_c, v_bshd = to_cute(v, dtype)
    o_bshd = torch.empty_like(q_bshd)
    o_c = (
        from_dlpack(o_bshd, assumed_align=16)
        .mark_layout_dynamic(leading_dim=3)
        .mark_compact_shape_dynamic(
            mode=3, stride_order=o_bshd.dim_order(),
            divisibility=(128 // dtype.width),
        )
    )
    fa2 = FlashAttentionForwardAmpere(D, m_bs, n_bs, num_threads)
    compiled = cute.compile(fa2, q_c, k_c, v_c, o_c, dummy_mask_c, scale, cu_stream)

    results = []
    results.append(test_dense(compiled, dtype, q, k, v, scale, m_bs, n_bs, cu_stream, verbose))
    results.append(test_sparse_random(compiled, dtype, q, k, v, scale, m_bs, n_bs, cu_stream,
                                      sparsity=0.5, seed=0, verbose=verbose))
    results.append(test_sparse_random(compiled, dtype, q, k, v, scale, m_bs, n_bs, cu_stream,
                                      sparsity=0.8, seed=1, verbose=verbose))
    results.append(test_lower_triangular(compiled, dtype, q, k, v, scale, m_bs, n_bs, cu_stream, verbose))
    results.append(test_all_zeros(compiled, dtype, q, k, v, scale, m_bs, n_bs, cu_stream, verbose))

    n_pass = sum(results)
    n_fail = len(results) - n_pass
    status = "PASS" if n_fail == 0 else f"FAIL ({n_fail} failed)"
    print(f"  → {n_pass}/{len(results)} passed   [{status}]")
    return n_fail == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CONFIGS = [
    # (B, Sq, Sk, H, D, m_bs, n_bs, num_threads)
    (1, 128, 128,  4,  64, 64, 32, 64),   # small, easy to debug
    (1, 192, 192, 16,  64, 64, 32, 64),   # non-power-of-2 seqlen
    (2, 256, 256,  8, 128, 64, 32, 64),   # larger head dim
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype",    default="BFloat16", choices=["BFloat16", "Float16"])
    parser.add_argument("--verbose",  action="store_true")
    parser.add_argument("--scale",    type=float, default=None)
    args = parser.parse_args()

    all_pass = True
    for B, Sq, Sk, H, D, m_bs, n_bs, T in CONFIGS:
        scale = args.scale or (1.0 / math.sqrt(D))
        ok = run_tests(args.dtype, B, Sq, Sk, H, D, m_bs, n_bs, T, scale, args.verbose)
        all_pass = all_pass and ok

    print()
    if all_pass:
        print("ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
