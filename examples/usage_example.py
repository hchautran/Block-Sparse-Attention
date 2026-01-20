"""
Complete Usage Example: Integrating Block Sparse Attention with ViT and SAM

This example shows how to:
1. Generate appropriate masks for your model
2. Prepare your Q, K, V tensors
3. Call block_sparse_attn_func correctly
4. Handle the output
"""

import torch
import sys
import os

# Add parent directory to path to import block_sparse_attn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from block_sparse_attn import block_sparse_attn_func
from vit_masks import generate_vit_spatial_locality_mask, generate_vit_hybrid_mask
from sam_masks import generate_sam_image_to_prompt_mask


def prepare_varlen_inputs(q, k, v, seqlen):
    """
    Convert standard (batch, seqlen, heads, dim) tensors to varlen format
    required by block_sparse_attn_func.

    Args:
        q, k, v: (batch_size, seqlen, num_heads, head_dim)
        seqlen: sequence length

    Returns:
        q_unpad, k_unpad, v_unpad: (total_tokens, num_heads, head_dim)
        cu_seqlens: cumulative sequence lengths
        max_seqlen: maximum sequence length
    """
    batch_size = q.shape[0]
    device = q.device

    # Reshape to (batch * seqlen, heads, dim)
    q_unpad = q.reshape(-1, q.shape[2], q.shape[3])
    k_unpad = k.reshape(-1, k.shape[2], k.shape[3])
    v_unpad = v.reshape(-1, v.shape[2], v.shape[3])

    # Create cumulative sequence lengths
    # For uniform sequence lengths: [0, seqlen, 2*seqlen, ..., batch*seqlen]
    cu_seqlens = torch.arange(
        0, (batch_size + 1) * seqlen, step=seqlen,
        dtype=torch.int32, device=device
    )

    max_seqlen = seqlen

    return q_unpad, k_unpad, v_unpad, cu_seqlens, max_seqlen


def vit_attention_example():
    """
    Example: Using block sparse attention with Vision Transformer
    """
    print("="*60)
    print("ViT Attention Example")
    print("="*60)

    # ViT-B/16 configuration
    img_size = 224
    patch_size = 16
    num_patches = (img_size // patch_size) ** 2  # 196
    num_tokens = num_patches + 1  # +1 for CLS token
    batch_size = 4
    num_heads = 12
    head_dim = 64
    block_size = 128

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16

    print(f"\nConfiguration:")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Patches: {num_patches} ({patch_size}x{patch_size})")
    print(f"  Total tokens: {num_tokens} (including CLS)")
    print(f"  Batch size: {batch_size}")
    print(f"  Heads: {num_heads}, Head dim: {head_dim}")
    print(f"  Device: {device}")

    # Create random Q, K, V tensors (simulating ViT attention layer)
    q = torch.randn(batch_size, num_tokens, num_heads, head_dim,
                    device=device, dtype=dtype)
    k = torch.randn(batch_size, num_tokens, num_heads, head_dim,
                    device=device, dtype=dtype)
    v = torch.randn(batch_size, num_tokens, num_heads, head_dim,
                    device=device, dtype=dtype)

    print(f"\nInput shapes: Q/K/V = {q.shape}")

    # Prepare inputs for block sparse attention
    q_unpad, k_unpad, v_unpad, cu_seqlens_q, max_seqlen_q = \
        prepare_varlen_inputs(q, k, v, num_tokens)

    cu_seqlens_k = cu_seqlens_q  # Same for self-attention
    max_seqlen_k = max_seqlen_q

    print(f"Varlen format: {q_unpad.shape}")
    print(f"cu_seqlens: {cu_seqlens_q}")

    # Generate spatial locality mask
    print("\n--- Using Spatial Locality Mask ---")
    base_blockmask = generate_vit_spatial_locality_mask(
        img_size=img_size,
        patch_size=patch_size,
        block_size=block_size,
        batch_size=batch_size,
        num_heads=num_heads,
        locality_radius=1,
        include_cls_token=True,
        device=device
    )

    print(f"Block mask shape: {base_blockmask.shape}")
    sparsity = 1 - base_blockmask.float().mean()
    print(f"Sparsity: {sparsity:.2%}")

    # Set up head mask type (all heads use block sparse)
    head_mask_type = torch.ones(num_heads, device=device, dtype=torch.int32)

    # Run block sparse attention
    output = block_sparse_attn_func(
        q_unpad, k_unpad, v_unpad,
        cu_seqlens_q, cu_seqlens_k,
        head_mask_type,
        streaming_info=None,
        base_blockmask=base_blockmask,
        max_seqlen_q_=max_seqlen_q,
        max_seqlen_k_=max_seqlen_k,
        p_dropout=0.0,
        deterministic=False,
        softmax_scale=None,
        is_causal=False,
        exact_streaming=False,
        return_attn_probs=False,
    )

    # Reshape output back to (batch, seqlen, heads, dim)
    output_reshaped = output.reshape(batch_size, num_tokens, num_heads, head_dim)

    print(f"Output shape: {output_reshaped.shape}")
    print("✓ Successfully computed sparse attention!\n")

    # Example 2: Hybrid attention (some dense, some sparse)
    print("--- Using Hybrid Mask (4 dense + 8 sparse heads) ---")

    base_blockmask_hybrid, head_mask_type_hybrid = generate_vit_hybrid_mask(
        img_size=img_size,
        patch_size=patch_size,
        block_size=block_size,
        batch_size=batch_size,
        num_heads=num_heads,
        num_global_heads=4,
        sparsity=0.5,
        include_cls_token=True,
        device=device
    )

    print(f"Block mask shape: {base_blockmask_hybrid.shape}")
    print(f"Head mask type: {head_mask_type_hybrid}")
    print(f"  0 = dense, 1 = sparse")

    output_hybrid = block_sparse_attn_func(
        q_unpad, k_unpad, v_unpad,
        cu_seqlens_q, cu_seqlens_k,
        head_mask_type_hybrid,
        streaming_info=None,
        base_blockmask=base_blockmask_hybrid,
        max_seqlen_q_=max_seqlen_q,
        max_seqlen_k_=max_seqlen_k,
        p_dropout=0.0,
        is_causal=False,
    )

    print(f"Output shape: {output_hybrid.reshape(batch_size, num_tokens, num_heads, head_dim).shape}")
    print("✓ Successfully computed hybrid attention!\n")


def sam_attention_example():
    """
    Example: Using block sparse attention with Segment Anything Model
    """
    print("="*60)
    print("SAM Attention Example")
    print("="*60)

    # SAM-B configuration
    img_size = 1024
    patch_size = 16
    num_image_tokens = (img_size // patch_size) ** 2  # 4096
    num_prompt_tokens = 5  # Example: point and box prompts
    total_tokens = num_image_tokens + num_prompt_tokens
    batch_size = 2
    num_heads = 12
    head_dim = 64
    block_size = 128

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16

    print(f"\nConfiguration:")
    print(f"  Image tokens: {num_image_tokens} ({img_size//patch_size}x{img_size//patch_size} patches)")
    print(f"  Prompt tokens: {num_prompt_tokens}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Batch size: {batch_size}")
    print(f"  Heads: {num_heads}, Head dim: {head_dim}")
    print(f"  Device: {device}")

    # Create Q, K, V tensors
    # In SAM, this would be after image encoder + prompt encoder
    q = torch.randn(batch_size, total_tokens, num_heads, head_dim,
                    device=device, dtype=dtype)
    k = torch.randn(batch_size, total_tokens, num_heads, head_dim,
                    device=device, dtype=dtype)
    v = torch.randn(batch_size, total_tokens, num_heads, head_dim,
                    device=device, dtype=dtype)

    # Prepare for block sparse attention
    q_unpad, k_unpad, v_unpad, cu_seqlens_q, max_seqlen_q = \
        prepare_varlen_inputs(q, k, v, total_tokens)
    cu_seqlens_k = cu_seqlens_q
    max_seqlen_k = max_seqlen_q

    print(f"\nInput shapes: Q/K/V = {q.shape}")

    # Generate SAM-specific mask
    print("\n--- Using Image-to-Prompt Mask ---")
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

    print(f"Block mask shape: {base_blockmask.shape}")
    sparsity = 1 - base_blockmask.float().mean()
    print(f"Overall sparsity: {sparsity:.2%}")
    print("Note: Image always attends to prompts (critical for conditioning)")

    # All heads use block sparse pattern
    head_mask_type = torch.ones(num_heads, device=device, dtype=torch.int32)

    # Run block sparse attention
    output = block_sparse_attn_func(
        q_unpad, k_unpad, v_unpad,
        cu_seqlens_q, cu_seqlens_k,
        head_mask_type,
        streaming_info=None,
        base_blockmask=base_blockmask,
        max_seqlen_q_=max_seqlen_q,
        max_seqlen_k_=max_seqlen_k,
        p_dropout=0.0,
        is_causal=False,
    )

    output_reshaped = output.reshape(batch_size, total_tokens, num_heads, head_dim)

    print(f"Output shape: {output_reshaped.shape}")
    print("✓ Successfully computed SAM sparse attention!\n")

    # Split output back to image and prompt tokens
    image_output = output_reshaped[:, :num_image_tokens]
    prompt_output = output_reshaped[:, num_image_tokens:]

    print(f"Image tokens output: {image_output.shape}")
    print(f"Prompt tokens output: {prompt_output.shape}")


def performance_comparison():
    """
    Compare performance of dense vs sparse attention
    """
    print("="*60)
    print("Performance Comparison")
    print("="*60)

    import time

    seq_len = 256
    batch_size = 8
    num_heads = 12
    head_dim = 64
    block_size = 128

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cpu':
        print("\nWarning: Running on CPU. Performance benefits minimal.")
        print("For real speedup, run on GPU!\n")
        return

    dtype = torch.float16

    # Create inputs
    q = torch.randn(batch_size, seq_len, num_heads, head_dim,
                    device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim,
                    device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim,
                    device=device, dtype=dtype)

    q_unpad, k_unpad, v_unpad, cu_seqlens, max_seqlen = \
        prepare_varlen_inputs(q, k, v, seq_len)

    print(f"Sequence length: {seq_len}")
    print(f"Batch size: {batch_size}")

    # Dense attention (sparsity = 0)
    print("\n1. Dense Attention (0% sparsity)")
    nrow = ncol = ((seq_len + block_size - 1) // block_size)
    dense_mask = torch.ones(batch_size, num_heads, nrow, ncol,
                           device=device, dtype=torch.bool)
    head_mask_type = torch.ones(num_heads, device=device, dtype=torch.int32)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = block_sparse_attn_func(
            q_unpad, k_unpad, v_unpad,
            cu_seqlens, cu_seqlens,
            head_mask_type, None, dense_mask,
            max_seqlen, max_seqlen,
            0.0, is_causal=False
        )
    torch.cuda.synchronize()
    dense_time = (time.time() - start) / 100

    print(f"   Average time: {dense_time*1000:.2f} ms")

    # Sparse attention (sparsity = 50%)
    print("\n2. Sparse Attention (50% sparsity)")
    sparse_mask = torch.zeros(batch_size, num_heads, nrow, ncol,
                             device=device, dtype=torch.bool)
    for b in range(batch_size):
        for h in range(num_heads):
            for i in range(nrow):
                selected = torch.randperm(ncol)[:ncol//2]
                sparse_mask[b, h, i, selected] = True

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = block_sparse_attn_func(
            q_unpad, k_unpad, v_unpad,
            cu_seqlens, cu_seqlens,
            head_mask_type, None, sparse_mask,
            max_seqlen, max_seqlen,
            0.0, is_causal=False
        )
    torch.cuda.synchronize()
    sparse_time = (time.time() - start) / 100

    print(f"   Average time: {sparse_time*1000:.2f} ms")
    print(f"\n   Speedup: {dense_time/sparse_time:.2f}x")

    # Very sparse attention (sparsity = 75%)
    print("\n3. Very Sparse Attention (75% sparsity)")
    very_sparse_mask = torch.zeros(batch_size, num_heads, nrow, ncol,
                                   device=device, dtype=torch.bool)
    for b in range(batch_size):
        for h in range(num_heads):
            for i in range(nrow):
                selected = torch.randperm(ncol)[:ncol//4]
                very_sparse_mask[b, h, i, selected] = True

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = block_sparse_attn_func(
            q_unpad, k_unpad, v_unpad,
            cu_seqlens, cu_seqlens,
            head_mask_type, None, very_sparse_mask,
            max_seqlen, max_seqlen,
            0.0, is_causal=False
        )
    torch.cuda.synchronize()
    very_sparse_time = (time.time() - start) / 100

    print(f"   Average time: {very_sparse_time*1000:.2f} ms")
    print(f"\n   Speedup: {dense_time/very_sparse_time:.2f}x")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Block Sparse Attention - Complete Usage Examples")
    print("="*60 + "\n")

    # Run examples
    vit_attention_example()
    print("\n")
    sam_attention_example()
    print("\n")
    performance_comparison()

    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60 + "\n")
