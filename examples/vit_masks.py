"""
Vision Transformer (ViT) Block Sparse Attention Mask Examples

This module provides various mask generation strategies for Vision Transformers
using block sparse attention patterns.
"""

import torch
import math


def round_to_multiple(x, base):
    """Round x up to the nearest multiple of base."""
    return ((x + base - 1) // base) * base


def generate_vit_spatial_locality_mask(
    img_size,
    patch_size,
    block_size,
    batch_size,
    num_heads,
    locality_radius=1,
    include_cls_token=True,
    device="cuda"
):
    """
    Generate a spatial locality mask for ViT where each patch primarily attends
    to nearby patches in the spatial grid.

    Args:
        img_size: Image size (assumes square images, e.g., 224)
        patch_size: Patch size (e.g., 16 for ViT-B/16)
        block_size: Block size for sparse attention (typically 128)
        batch_size: Batch size
        num_heads: Number of attention heads
        locality_radius: How many patches away to attend (1 = immediate neighbors)
        include_cls_token: Whether to include CLS token (always attends globally)
        device: Device to create tensors on

    Returns:
        base_blockmask: (batch_size, num_heads, nrow, ncol) boolean mask
    """
    # Calculate number of patches
    num_patches = (img_size // patch_size) ** 2
    seq_len = num_patches + (1 if include_cls_token else 0)

    # Calculate grid dimensions
    grid_size = img_size // patch_size

    # Calculate block dimensions
    nrow = ncol = round_to_multiple(seq_len, block_size) // block_size

    # Create attention mask at token level first
    token_mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)

    # CLS token attends to everything and everything attends to CLS
    if include_cls_token:
        token_mask[0, :] = True  # CLS attends to all
        token_mask[:, 0] = True  # All attend to CLS
        offset = 1
    else:
        offset = 0

    # Spatial locality for patch tokens
    for i in range(num_patches):
        row_i = i // grid_size
        col_i = i % grid_size

        for j in range(num_patches):
            row_j = j // grid_size
            col_j = j % grid_size

            # Manhattan distance or Euclidean distance
            distance = abs(row_i - row_j) + abs(col_i - col_j)

            if distance <= locality_radius * 2:  # 2x for Manhattan distance
                token_mask[i + offset, j + offset] = True

    # Convert to block mask
    token_mask_padded = torch.nn.functional.pad(
        token_mask,
        (0, round_to_multiple(seq_len, block_size) - seq_len,
         0, round_to_multiple(seq_len, block_size) - seq_len)
    )

    # Block mask: True if ANY token in the block pair should attend
    block_mask = torch.zeros(nrow, ncol, device=device, dtype=torch.bool)
    for i in range(nrow):
        for j in range(ncol):
            block_mask[i, j] = token_mask_padded[
                i*block_size:(i+1)*block_size,
                j*block_size:(j+1)*block_size
            ].any()

    # Replicate for batch and heads
    base_blockmask = block_mask.unsqueeze(0).unsqueeze(0).expand(
        batch_size, num_heads, nrow, ncol
    ).contiguous()

    return base_blockmask


def generate_vit_random_sparse_mask(
    img_size,
    patch_size,
    block_size,
    batch_size,
    num_heads,
    sparsity=0.5,
    include_cls_token=True,
    device="cuda"
):
    """
    Generate random sparse mask for ViT with guaranteed CLS token attention.

    Args:
        img_size: Image size
        patch_size: Patch size
        block_size: Block size for sparse attention
        batch_size: Batch size
        num_heads: Number of attention heads
        sparsity: Fraction of blocks to skip (0.0 = dense, 1.0 = maximum sparse)
        include_cls_token: Whether to include CLS token
        device: Device to create tensors on

    Returns:
        base_blockmask: (batch_size, num_heads, nrow, ncol) boolean mask
    """
    num_patches = (img_size // patch_size) ** 2
    seq_len = num_patches + (1 if include_cls_token else 0)

    nrow = ncol = round_to_multiple(seq_len, block_size) // block_size

    base_mask = torch.zeros(batch_size, num_heads, nrow, ncol, device=device, dtype=torch.bool)

    density = 1.0 - sparsity

    for batch in range(batch_size):
        for head in range(num_heads):
            # Random sparse pattern
            for i in range(nrow):
                if density < 1.0:
                    num_blocks_to_keep = max(1, int(density * ncol))
                    selected_cols = torch.randperm(ncol)[:num_blocks_to_keep]
                    base_mask[batch, head, i, selected_cols] = True
                else:
                    base_mask[batch, head, i, :] = True

            # Always ensure first block (CLS) attends globally
            if include_cls_token and nrow > 0:
                base_mask[batch, head, 0, :] = True  # CLS attends to all
                base_mask[batch, head, :, 0] = True  # All attend to CLS

    return base_mask


def generate_vit_strided_mask(
    img_size,
    patch_size,
    block_size,
    batch_size,
    num_heads,
    stride=2,
    include_cls_token=True,
    device="cuda"
):
    """
    Generate a strided attention mask where each patch attends to every Nth patch.
    Useful for reducing computation while maintaining long-range dependencies.

    Args:
        img_size: Image size
        patch_size: Patch size
        block_size: Block size for sparse attention
        batch_size: Batch size
        num_heads: Number of attention heads
        stride: Attention stride (2 = every other patch, 3 = every third, etc.)
        include_cls_token: Whether to include CLS token
        device: Device to create tensors on

    Returns:
        base_blockmask: (batch_size, num_heads, nrow, ncol) boolean mask
    """
    num_patches = (img_size // patch_size) ** 2
    seq_len = num_patches + (1 if include_cls_token else 0)

    nrow = ncol = round_to_multiple(seq_len, block_size) // block_size

    base_mask = torch.zeros(batch_size, num_heads, nrow, ncol, device=device, dtype=torch.bool)

    for batch in range(batch_size):
        for head in range(num_heads):
            # Strided pattern
            for i in range(nrow):
                base_mask[batch, head, i, ::stride] = True

            # CLS token special handling
            if include_cls_token:
                base_mask[batch, head, 0, :] = True
                base_mask[batch, head, :, 0] = True

    return base_mask


def generate_vit_hybrid_mask(
    img_size,
    patch_size,
    block_size,
    batch_size,
    num_heads,
    num_global_heads=4,
    sparsity=0.5,
    include_cls_token=True,
    device="cuda"
):
    """
    Generate a hybrid mask where some heads use dense attention and others use sparse.
    This follows patterns from DuoAttention and similar work.

    Args:
        img_size: Image size
        patch_size: Patch size
        block_size: Block size for sparse attention
        batch_size: Batch size
        num_heads: Total number of attention heads
        num_global_heads: Number of heads with dense (global) attention
        sparsity: Sparsity for sparse heads
        include_cls_token: Whether to include CLS token
        device: Device to create tensors on

    Returns:
        base_blockmask: (batch_size, num_heads, nrow, ncol) boolean mask
        head_mask_type: (num_heads,) tensor indicating mask type per head
    """
    num_patches = (img_size // patch_size) ** 2
    seq_len = num_patches + (1 if include_cls_token else 0)

    nrow = ncol = round_to_multiple(seq_len, block_size) // block_size

    # Only create masks for sparse heads
    num_sparse_heads = num_heads - num_global_heads
    base_mask = torch.zeros(batch_size, num_sparse_heads, nrow, ncol, device=device, dtype=torch.bool)

    # Generate sparse masks
    density = 1.0 - sparsity
    for batch in range(batch_size):
        for head in range(num_sparse_heads):
            for i in range(nrow):
                num_blocks_to_keep = max(1, int(density * ncol))
                selected_cols = torch.randperm(ncol)[:num_blocks_to_keep]
                base_mask[batch, head, i, selected_cols] = True

            if include_cls_token:
                base_mask[batch, head, 0, :] = True
                base_mask[batch, head, :, 0] = True

    # Head mask type: 0 = dense, 1 = sparse
    head_mask_type = torch.zeros(num_heads, device=device, dtype=torch.int32)
    head_mask_type[:num_global_heads] = 0  # Dense heads
    head_mask_type[num_global_heads:] = 1  # Sparse heads

    return base_mask, head_mask_type


def generate_vit_multiscale_mask(
    img_size,
    patch_size,
    block_size,
    batch_size,
    num_heads,
    include_cls_token=True,
    device="cuda"
):
    """
    Generate multi-scale attention mask where different heads attend at different
    spatial scales. Early heads focus on local patterns, later heads on global patterns.

    Args:
        img_size: Image size
        patch_size: Patch size
        block_size: Block size for sparse attention
        batch_size: Batch size
        num_heads: Number of attention heads
        include_cls_token: Whether to include CLS token
        device: Device to create tensors on

    Returns:
        base_blockmask: (batch_size, num_heads, nrow, ncol) boolean mask
    """
    num_patches = (img_size // patch_size) ** 2
    seq_len = num_patches + (1 if include_cls_token else 0)
    grid_size = img_size // patch_size

    nrow = ncol = round_to_multiple(seq_len, block_size) // block_size

    base_mask = torch.zeros(batch_size, num_heads, nrow, ncol, device=device, dtype=torch.bool)

    # Create token-level masks for each head with different radii
    for head in range(num_heads):
        # Scale radius from 1 (local) to grid_size (global)
        radius = 1 + (head * (grid_size - 1)) // (num_heads - 1) if num_heads > 1 else grid_size

        token_mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)

        if include_cls_token:
            token_mask[0, :] = True
            token_mask[:, 0] = True
            offset = 1
        else:
            offset = 0

        # Spatial locality at this scale
        for i in range(num_patches):
            row_i = i // grid_size
            col_i = i % grid_size

            for j in range(num_patches):
                row_j = j // grid_size
                col_j = j % grid_size

                distance = abs(row_i - row_j) + abs(col_i - col_j)
                if distance <= radius * 2:
                    token_mask[i + offset, j + offset] = True

        # Convert to block mask
        token_mask_padded = torch.nn.functional.pad(
            token_mask,
            (0, round_to_multiple(seq_len, block_size) - seq_len,
             0, round_to_multiple(seq_len, block_size) - seq_len)
        )

        for i in range(nrow):
            for j in range(ncol):
                base_mask[:, head, i, j] = token_mask_padded[
                    i*block_size:(i+1)*block_size,
                    j*block_size:(j+1)*block_size
                ].any()

    return base_mask


# Example usage
if __name__ == "__main__":
    # ViT-B/16 on 224x224 images
    img_size = 224
    patch_size = 16
    block_size = 128
    batch_size = 2
    num_heads = 12

    print("=== ViT Block Sparse Attention Mask Examples ===\n")

    # Example 1: Spatial Locality
    print("1. Spatial Locality Mask (radius=1)")
    mask = generate_vit_spatial_locality_mask(
        img_size, patch_size, block_size, batch_size, num_heads, locality_radius=1
    )
    print(f"   Shape: {mask.shape}")
    print(f"   Sparsity: {1 - mask.float().mean():.2%}")
    print(f"   Blocks computed per row: {mask[0, 0].sum(dim=1).float().mean():.1f}/{mask.shape[-1]}\n")

    # Example 2: Random Sparse
    print("2. Random Sparse Mask (50% sparsity)")
    mask = generate_vit_random_sparse_mask(
        img_size, patch_size, block_size, batch_size, num_heads, sparsity=0.5
    )
    print(f"   Shape: {mask.shape}")
    print(f"   Sparsity: {1 - mask.float().mean():.2%}\n")

    # Example 3: Strided
    print("3. Strided Mask (stride=2)")
    mask = generate_vit_strided_mask(
        img_size, patch_size, block_size, batch_size, num_heads, stride=2
    )
    print(f"   Shape: {mask.shape}")
    print(f"   Sparsity: {1 - mask.float().mean():.2%}\n")

    # Example 4: Hybrid
    print("4. Hybrid Mask (4 dense heads + 8 sparse heads)")
    mask, head_types = generate_vit_hybrid_mask(
        img_size, patch_size, block_size, batch_size, num_heads, num_global_heads=4
    )
    print(f"   Sparse mask shape: {mask.shape}")
    print(f"   Head types: {head_types}")
    print(f"   Note: Dense heads (type 0) don't need masks\n")

    # Example 5: Multi-scale
    print("5. Multi-scale Mask (different radius per head)")
    mask = generate_vit_multiscale_mask(
        img_size, patch_size, block_size, batch_size, num_heads
    )
    print(f"   Shape: {mask.shape}")
    for head in [0, num_heads//2, num_heads-1]:
        sparsity = 1 - mask[0, head].float().mean()
        print(f"   Head {head} sparsity: {sparsity:.2%}")
