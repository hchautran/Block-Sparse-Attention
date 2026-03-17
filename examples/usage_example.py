"""
SAM Block Sparse Attention - Usage Example

This example shows how to use the simplified SAM-only interface.

Demonstrates:
1. Simple interface (automatic varlen conversion)
2. Advanced interface (manual varlen conversion)
3. Different mask patterns for SAM
4. Performance comparison
"""
import torch
import torch.nn.functional as F
from typing import Tuple

from block_sparse_attn.attention import block_sparse_attn_simple
import time


batch_size = 400 
seq_len = 196 
num_heads =  1 
head_dim = 64
block_size = 128
dtype = torch.float16


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]



def get_decomposed_rel_pos(
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
    pos = (
        rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).reshape(B, q_h * q_w, k_h * k_w)

    return pos



def naive_sam_attn(x, rel_pos_h:torch.Tensor, rel_pos_w:torch.Tensor, scale:float, qkv: torch.nn.Linear):
    """Naive SAM attention implementation for correctness check."""
    B, H, W, _ = x.shape
    qkv = qkv(x).reshape(B, H * W, 3, num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.reshape(3, B * num_heads, H * W, -1).unbind(0)
    attn = (q * scale) @ k.transpose(-2, -1)

    pos = get_decomposed_rel_pos(q, rel_pos_h, rel_pos_w, (H, W), (H, W))
    attn = (attn + pos).softmax(dim=-1)
    x = (attn @ v).view(B, num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
    return x


def flash_sam_attn(x, rel_pos_h:torch.Tensor, rel_pos_w:torch.Tensor, scale:float, qkv: torch.nn.Linear, dense_mask: torch.Tensor, head_mask_type: torch.Tensor):
    B, H, W, _ = x.shape
    qkv = qkv(x).reshape(B, H * W, 3, num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.reshape(3, B * num_heads, H * W, -1).unbind(0)
    pos = get_decomposed_rel_pos(q, rel_pos_h, rel_pos_w, (H, W), (H, W))
    head_mask_type[::2] = 1
    x = block_sparse_attn_simple(
        q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-2),
        dense_mask, 
        positional=pos.unsqueeze(1), 
        softmax_scale=scale,
        head_mask_type=head_mask_type
    ).reshape(B, H, W, -1)
    return x

     

def example_3_correctness():
    """Example 3: Correctness check vs dense attention"""
    print("\n" + "="*70)
    print("Example 3: Correctness Check (Dense vs Block Sparse)")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("\n⚠ Warning: CUDA not available. Skipping.")
        return

    H, W = int(seq_len**0.5), int(seq_len**0.5)
    x = torch.randn(batch_size, H, W, head_dim, device=device, dtype=dtype)
    qkv = torch.nn.Linear(head_dim, 3 * head_dim, dtype=dtype).to(device)
    rel_pos_h = torch.randn(2 * H - 1, head_dim, device=device, dtype=dtype)
    rel_pos_w = torch.randn(2 * W - 1, head_dim, device=device, dtype=dtype)
    scale = head_dim ** -0.5
    nrow = ncol = (seq_len + block_size - 1) // block_size
    dense_mask = torch.ones(batch_size, num_heads, nrow, ncol, device=device, dtype=torch.bool)
    head_mask_type = 1 * torch.ones(batch_size, dtype=torch.int32, device=device)
    
    out_dense = naive_sam_attn(x, rel_pos_h, rel_pos_w, scale, qkv)
    out_sparse = flash_sam_attn(x, rel_pos_h, rel_pos_w, scale, qkv, dense_mask, head_mask_type)
    print(out_sparse.shape, out_dense.shape)

    diff = (out_sparse - out_dense).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    print(f"\nMax abs diff: {max_abs:.3e}")
    print(f"Mean abs diff: {mean_abs:.3e}")

    try:
        torch.testing.assert_close(out_sparse, out_dense, rtol=1e-2, atol=1e-2)
        print("✅ Correctness check passed.")
    except AssertionError as exc:
        print("❌ Correctness check failed.")
        print(str(exc))

def diagonal_band(n, k=1, device="cpu"):
    idx = torch.arange(n, device=device)
    dist = (idx[:, None] - idx[None, :]).abs()
    return (dist <= k).int()

def example_4_performance():
    """Example 4: Performance comparison"""
    print("\n" + "="*70)
    print("Example 4: Performance Comparison")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("\n⚠️  Warning: CUDA not available. Skipping performance test.")
        return

    print(f"\nSetup:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Batch size: {batch_size}")
    print(f"  Heads: {num_heads}")

    nrow = ncol = (seq_len + block_size - 1) // block_size
    dtype = torch.float16
    # Dense attention (0% sparsity)
    print("\n1. Dense Attention (0% sparsity)")
    dense_mask = torch.ones(batch_size, num_heads, nrow, ncol, device=device, dtype=torch.bool)
    sparse_mask = diagonal_band(nrow, k=int(0.5* nrow), device=device)[None, None,...].expand(dense_mask.shape)
    very_sparse_mask = diagonal_band(nrow, k=int(0.25* nrow), device=device)[None, None,...].expand(dense_mask.shape)

  


    # SAM-style attention (naive vs flash)
    H_sam = W_sam = int(seq_len ** 0.5)
    x_sam = torch.randn(batch_size, H_sam, W_sam, head_dim, device=device, dtype=dtype)
    qkv_linear = torch.nn.Linear(head_dim, 3 * head_dim, dtype=dtype).to(device)
    rel_pos_h = torch.randn(2 * H_sam - 1, head_dim, device=device, dtype=dtype)
    rel_pos_w = torch.randn(2 * W_sam - 1, head_dim, device=device, dtype=dtype)
    head_mask_type = torch.ones(batch_size, dtype=torch.int32, device=device)

    print("\n4. SAM Attention (naive, with rel pos)")
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = naive_sam_attn(x_sam, rel_pos_h, rel_pos_w, scale=head_dim**-0.5, qkv=qkv_linear)
    torch.cuda.synchronize()
    naive_time = (time.time() - start) / 100
    print(f"   Time: {naive_time*1000:.2f} ms")

    print("\n5. SAM Attention (flash sparse dense mask, with rel pos)")
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = flash_sam_attn(x_sam, rel_pos_h, rel_pos_w, scale=head_dim**-0.5, qkv=qkv_linear,
                           dense_mask=dense_mask, head_mask_type=head_mask_type.clone())
    torch.cuda.synchronize()
    flash_dense_time = (time.time() - start) / 100
    print(f"   Time: {flash_dense_time*1000:.2f} ms")
    print(f"   Speedup vs naive: {naive_time/flash_dense_time:.2f}x")

    print("\n6. SAM Attention (flash sparse 50% mask, with rel pos)")
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = flash_sam_attn(x_sam, rel_pos_h, rel_pos_w, scale=head_dim**-0.5, qkv=qkv_linear,
                           dense_mask=sparse_mask, head_mask_type=head_mask_type.clone())
    torch.cuda.synchronize()
    flash_sparse_time = (time.time() - start) / 100
    print(f"   Time: {flash_sparse_time*1000:.2f} ms")
    print(f"   Speedup vs naive: {naive_time/flash_sparse_time:.2f}x")


if __name__ == "__main__":


    example_3_correctness()
    example_4_performance()

