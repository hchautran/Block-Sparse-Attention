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
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from block_sparse_attn.sam_attention import sam_block_sparse_attn_simple
from examples.sam_masks import generate_sam_image_to_prompt_mask


def example_1_simple_interface():
    """Example 1: Simple interface with automatic varlen conversion"""
    print("="*70)
    print("Example 1: Simple Interface (Recommended for most users)")
    print("="*70)

    # SAM-B configuration
    img_size = 1024
    patch_size = 16
    num_image_tokens = (img_size // patch_size) ** 2  # 4096
    num_prompt_tokens = 5
    total_tokens = num_image_tokens + num_prompt_tokens

    batch_size = 2
    num_heads = 12
    head_dim = 64
    block_size = 128

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16

    print(f"\nConfiguration:")
    print(f"  Image tokens: {num_image_tokens} ({img_size//patch_size}x{img_size//patch_size})")
    print(f"  Prompt tokens: {num_prompt_tokens}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Batch size: {batch_size}")
    print(f"  Heads: {num_heads}, Head dim: {head_dim}")
    print(f"  Device: {device}")

    if device == 'cpu':
        print("\n⚠️  Warning: CUDA not available. This example requires a GPU.")
        return

    # Create Q, K, V tensors in standard format (batch, seqlen, heads, dim)
    q = torch.randn(batch_size, total_tokens, num_heads, head_dim,
                    device=device, dtype=dtype)
    k = torch.randn(batch_size, total_tokens, num_heads, head_dim,
                    device=device, dtype=dtype)
    v = torch.randn(batch_size, total_tokens, num_heads, head_dim,
                    device=device, dtype=dtype)

    print(f"\nInput shape: {q.shape}")

    # Generate SAM-specific sparse mask
    base_blockmask = generate_sam_image_to_prompt_mask(
        num_image_tokens=num_image_tokens,
        num_prompt_tokens=num_prompt_tokens,
        block_size=block_size,
        batch_size=batch_size,
        num_heads=num_heads,
        sparse_image_to_image=True,
        image_sparsity=0.6,
        device=device
    )

    sparsity = 1 - base_blockmask.float().mean()
    print(f"\nBlock mask shape: {base_blockmask.shape}")
    print(f"Sparsity: {sparsity:.2%}")

    # Use simple interface - it handles varlen conversion automatically!
    output = sam_block_sparse_attn_simple(
        q, k, v,
        base_blockmask=base_blockmask
    )

    print(f"\nOutput shape: {output.shape}")
    print("✅ Success! Output matches input shape.")

    # Split into image and prompt outputs
    image_output = output[:, :num_image_tokens]
    prompt_output = output[:, num_image_tokens:]
    print(f"\nImage output: {image_output.shape}")
    print(f"Prompt output: {prompt_output.shape}")


def example_2_advanced_interface():
    """Example 2: Advanced interface with manual varlen conversion"""
    print("\n" + "="*70)
    print("Example 2: Advanced Interface (Manual varlen conversion)")
    print("="*70)

    from block_sparse_attn.sam_attention import sam_block_sparse_attn, prepare_varlen_inputs

    # Smaller example for clarity
    num_tokens = 256
    batch_size = 4
    num_heads = 8
    head_dim = 64
    block_size = 128

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("\n⚠️  Warning: CUDA not available. Skipping.")
        return

    dtype = torch.float16

    print(f"\nConfiguration:")
    print(f"  Tokens: {num_tokens}")
    print(f"  Batch size: {batch_size}")
    print(f"  Heads: {num_heads}, Head dim: {head_dim}")

    # Create tensors
    q = torch.randn(batch_size, num_tokens, num_heads, head_dim,
                    device=device, dtype=dtype)
    k = torch.randn(batch_size, num_tokens, num_heads, head_dim,
                    device=device, dtype=dtype)
    v = torch.randn(batch_size, num_tokens, num_heads, head_dim,
                    device=device, dtype=dtype)

    # Manual varlen conversion
    q_unpad, k_unpad, v_unpad, cu_seqlens, max_seqlen = prepare_varlen_inputs(
        q, k, v, num_tokens
    )

    print(f"\nOriginal shape: {q.shape}")
    print(f"Varlen shape: {q_unpad.shape}")
    print(f"cu_seqlens: {cu_seqlens}")

    # Create a simple block mask (50% sparse)
    nrow = ncol = (num_tokens + block_size - 1) // block_size
    base_blockmask = torch.zeros(batch_size, num_heads, nrow, ncol,
                                  device=device, dtype=torch.bool)
    for b in range(batch_size):
        for h in range(num_heads):
            for i in range(nrow):
                # Keep 50% of blocks
                selected = torch.randperm(ncol, device=device)[:ncol//2]
                base_blockmask[b, h, i, selected] = True

    # Head mask type (all use sparse)
    head_mask_type = torch.ones(num_heads, device=device, dtype=torch.int32)

    # Advanced interface with full control
    output = sam_block_sparse_attn(
        q_unpad, k_unpad, v_unpad,
        cu_seqlens, cu_seqlens,
        head_mask_type,
        base_blockmask,
        max_seqlen, max_seqlen,
        softmax_scale=None
    )

    # Reshape back
    output = output.reshape(batch_size, num_tokens, num_heads, head_dim)

    print(f"\nOutput shape: {output.shape}")
    print("✅ Success!")


def example_3_different_masks():
    """Example 3: Different mask patterns"""
    print("\n" + "="*70)
    print("Example 3: Different Mask Patterns")
    print("="*70)

    from examples.sam_masks import (
        generate_sam_hierarchical_mask,
        generate_sam_hybrid_encoder_mask
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("\n⚠️  Warning: CUDA not available. Skipping.")
        return

    num_image_tokens = 1024
    num_prompt_tokens = 5
    total_tokens = num_image_tokens + num_prompt_tokens
    batch_size = 2
    num_heads = 12
    head_dim = 64
    block_size = 128

    q = torch.randn(batch_size, total_tokens, num_heads, head_dim,
                    device=device, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Pattern 1: Hierarchical (different scales per head)
    print("\n1. Hierarchical Mask (multi-scale)")
    mask = generate_sam_hierarchical_mask(
        num_image_tokens, num_prompt_tokens, block_size,
        batch_size, num_heads, device=device
    )
    output = sam_block_sparse_attn_simple(q, k, v, mask)
    print(f"   Sparsity: {1 - mask.float().mean():.2%}")
    print(f"   Output shape: {output.shape}")

    # Pattern 2: Hybrid (some dense, some sparse heads)
    print("\n2. Hybrid Mask (4 dense + 8 sparse heads)")
    mask, head_types = generate_sam_hybrid_encoder_mask(
        num_image_tokens, block_size, batch_size, num_heads,
        num_dense_heads=4, sparsity=0.7, device=device
    )
    # For hybrid, we only pass the sparse mask
    output = sam_block_sparse_attn_simple(q[:, :num_image_tokens],
                                         k[:, :num_image_tokens],
                                         v[:, :num_image_tokens],
                                         mask, head_types)
    print(f"   Head types: {head_types}")
    print(f"   Sparse heads sparsity: {1 - mask.float().mean():.2%}")
    print(f"   Output shape: {output.shape}")


def example_4_performance():
    """Example 4: Performance comparison"""
    print("\n" + "="*70)
    print("Example 4: Performance Comparison")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("\n⚠️  Warning: CUDA not available. Skipping performance test.")
        return

    import time

    seq_len = 4096
    batch_size = 8
    num_heads = 12
    head_dim = 64
    block_size = 128

    q = torch.randn(batch_size, seq_len, num_heads, head_dim,
                    device=device, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    print(f"\nSetup:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Batch size: {batch_size}")
    print(f"  Heads: {num_heads}")

    nrow = ncol = (seq_len + block_size - 1) // block_size

    # Dense attention (0% sparsity)
    print("\n1. Dense Attention (0% sparsity)")
    dense_mask = torch.ones(batch_size, num_heads, nrow, ncol,
                           device=device, dtype=torch.bool)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = sam_block_sparse_attn_simple(q, k, v, dense_mask)
    torch.cuda.synchronize()
    dense_time = (time.time() - start) / 100
    print(f"   Time: {dense_time*1000:.2f} ms")

    # 50% sparse
    print("\n2. Sparse Attention (50% sparsity)")
    sparse_mask = torch.zeros(batch_size, num_heads, nrow, ncol,
                              device=device, dtype=torch.bool)
    for b in range(batch_size):
        for h in range(num_heads):
            for i in range(nrow):
                selected = torch.randperm(ncol, device=device)[:ncol//2]
                sparse_mask[b, h, i, selected] = True

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = sam_block_sparse_attn_simple(q, k, v, sparse_mask)
    torch.cuda.synchronize()
    sparse_time = (time.time() - start) / 100
    print(f"   Time: {sparse_time*1000:.2f} ms")
    print(f"   Speedup: {dense_time/sparse_time:.2f}x")

    # 75% sparse
    print("\n3. Very Sparse Attention (75% sparsity)")
    very_sparse_mask = torch.zeros(batch_size, num_heads, nrow, ncol,
                                   device=device, dtype=torch.bool)
    for b in range(batch_size):
        for h in range(num_heads):
            for i in range(nrow):
                selected = torch.randperm(ncol, device=device)[:ncol//4]
                very_sparse_mask[b, h, i, selected] = True

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = sam_block_sparse_attn_simple(q, k, v, very_sparse_mask)
    torch.cuda.synchronize()
    very_sparse_time = (time.time() - start) / 100
    print(f"   Time: {very_sparse_time*1000:.2f} ms")
    print(f"   Speedup: {dense_time/very_sparse_time:.2f}x")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("SAM Block Sparse Attention - Usage Examples")
    print("="*70)

    example_1_simple_interface()
    example_2_advanced_interface()
    example_3_different_masks()
    example_4_performance()

    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70 + "\n")
