"""
Segment Anything Model (SAM) Block Sparse Attention Mask Examples

SAM has a unique architecture with:
- Image encoder (ViT-based) that processes image patches
- Prompt encoder that processes points, boxes, or mask prompts
- Mask decoder with cross-attention between image embeddings and prompts

This module provides mask generation strategies for SAM's attention patterns.
"""

import torch
import math


def round_to_multiple(x, base):
    """Round x up to the nearest multiple of base."""
    return ((x + base - 1) // base) * base


def generate_sam_image_to_prompt_mask(
    num_image_tokens,
    num_prompt_tokens,
    block_size,
    batch_size,
    num_heads,
    sparse_image_to_image=True,
    image_sparsity=0.5,
    device="cuda"
):
    """
    Generate mask for SAM where:
    - Image tokens ALWAYS attend to prompt tokens (critical for conditioning)
    - Prompt tokens ALWAYS attend to image tokens (need full image context)
    - Image tokens can use sparse attention to other image tokens (optional)

    Args:
        num_image_tokens: Number of image tokens (e.g., 64x64 = 4096 for SAM-B)
        num_prompt_tokens: Number of prompt tokens (variable, typically 2-10)
        block_size: Block size for sparse attention
        batch_size: Batch size
        num_heads: Number of attention heads
        sparse_image_to_image: If True, image-to-image uses sparse pattern
        image_sparsity: Sparsity for image-to-image attention
        device: Device to create tensors on

    Returns:
        base_blockmask: (batch_size, num_heads, nrow, ncol) boolean mask
    """
    total_tokens = num_image_tokens + num_prompt_tokens
    nrow = ncol = round_to_multiple(total_tokens, block_size) // block_size

    # Create token-level mask
    token_mask = torch.zeros(total_tokens, total_tokens, device=device, dtype=torch.bool)

    # Image tokens (0 to num_image_tokens-1)
    # Prompt tokens (num_image_tokens to total_tokens-1)

    # 1. Prompt tokens attend to ALL image tokens (need full context)
    token_mask[num_image_tokens:, :num_image_tokens] = True

    # 2. Prompt tokens attend to all other prompt tokens
    token_mask[num_image_tokens:, num_image_tokens:] = True

    # 3. Image tokens ALWAYS attend to ALL prompt tokens (conditioning)
    token_mask[:num_image_tokens, num_image_tokens:] = True

    # 4. Image-to-image attention (can be sparse or dense)
    if sparse_image_to_image:
        # Use spatial locality for image tokens
        grid_size = int(math.sqrt(num_image_tokens))
        for i in range(num_image_tokens):
            row_i = i // grid_size
            col_i = i % grid_size

            # Random sparse pattern with spatial bias
            num_neighbors = max(1, int((1 - image_sparsity) * num_image_tokens))

            # Prioritize spatial neighbors
            neighbors = []
            for j in range(num_image_tokens):
                row_j = j // grid_size
                col_j = j % grid_size
                distance = abs(row_i - row_j) + abs(col_i - col_j)
                neighbors.append((distance, j))

            neighbors.sort()
            selected = [idx for _, idx in neighbors[:num_neighbors]]
            token_mask[i, selected] = True
    else:
        # Dense image-to-image
        token_mask[:num_image_tokens, :num_image_tokens] = True

    # Convert to block mask
    token_mask_padded = torch.nn.functional.pad(
        token_mask,
        (0, round_to_multiple(total_tokens, block_size) - total_tokens,
         0, round_to_multiple(total_tokens, block_size) - total_tokens)
    )

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


def generate_sam_hierarchical_mask(
    num_image_tokens,
    num_prompt_tokens,
    block_size,
    batch_size,
    num_heads,
    device="cuda"
):
    """
    Generate hierarchical attention mask for SAM where different heads use
    different patterns:
    - Some heads: dense attention (global context)
    - Some heads: local spatial attention (fine details)
    - All heads: full prompt attention (conditioning)

    Args:
        num_image_tokens: Number of image tokens
        num_prompt_tokens: Number of prompt tokens
        block_size: Block size for sparse attention
        batch_size: Batch size
        num_heads: Number of attention heads
        device: Device to create tensors on

    Returns:
        base_blockmask: (batch_size, num_heads, nrow, ncol) boolean mask
    """
    total_tokens = num_image_tokens + num_prompt_tokens
    nrow = ncol = round_to_multiple(total_tokens, block_size) // block_size

    base_mask = torch.zeros(batch_size, num_heads, nrow, ncol, device=device, dtype=torch.bool)

    grid_size = int(math.sqrt(num_image_tokens))

    for head in range(num_heads):
        # Determine attention radius for this head
        # Early heads: local, later heads: more global
        radius = 1 + (head * grid_size) // num_heads

        token_mask = torch.zeros(total_tokens, total_tokens, device=device, dtype=torch.bool)

        # Prompts attend to everything
        token_mask[num_image_tokens:, :] = True

        # Image tokens attend to all prompts
        token_mask[:num_image_tokens, num_image_tokens:] = True

        # Image-to-image with varying locality
        for i in range(num_image_tokens):
            row_i = i // grid_size
            col_i = i % grid_size

            for j in range(num_image_tokens):
                row_j = j // grid_size
                col_j = j % grid_size

                distance = abs(row_i - row_j) + abs(col_i - col_j)
                if distance <= radius:
                    token_mask[i, j] = True

        # Convert to block mask
        token_mask_padded = torch.nn.functional.pad(
            token_mask,
            (0, round_to_multiple(total_tokens, block_size) - total_tokens,
             0, round_to_multiple(total_tokens, block_size) - total_tokens)
        )

        for i in range(nrow):
            for j in range(ncol):
                base_mask[:, head, i, j] = token_mask_padded[
                    i*block_size:(i+1)*block_size,
                    j*block_size:(j+1)*block_size
                ].any()

    return base_mask


def generate_sam_region_focused_mask(
    num_image_tokens,
    num_prompt_tokens,
    prompt_positions,
    block_size,
    batch_size,
    num_heads,
    focus_radius=8,
    device="cuda"
):
    """
    Generate mask that focuses attention around prompt regions.
    Useful when prompts indicate specific regions of interest.

    Args:
        num_image_tokens: Number of image tokens
        num_prompt_tokens: Number of prompt tokens
        prompt_positions: List of (row, col) positions for prompts in image grid
        block_size: Block size for sparse attention
        batch_size: Batch size
        num_heads: Number of attention heads
        focus_radius: Radius around prompts to attend densely
        device: Device to create tensors on

    Returns:
        base_blockmask: (batch_size, num_heads, nrow, ncol) boolean mask
    """
    total_tokens = num_image_tokens + num_prompt_tokens
    nrow = ncol = round_to_multiple(total_tokens, block_size) // block_size

    grid_size = int(math.sqrt(num_image_tokens))

    token_mask = torch.zeros(total_tokens, total_tokens, device=device, dtype=torch.bool)

    # Prompts always attend to everything
    token_mask[num_image_tokens:, :] = True

    # Image tokens always attend to prompts
    token_mask[:num_image_tokens, num_image_tokens:] = True

    # Image-to-image: dense around prompt regions, sparse elsewhere
    for i in range(num_image_tokens):
        row_i = i // grid_size
        col_i = i % grid_size

        # Check if this token is near any prompt
        near_prompt = False
        for prompt_row, prompt_col in prompt_positions:
            distance = abs(row_i - prompt_row) + abs(col_i - prompt_col)
            if distance <= focus_radius:
                near_prompt = True
                break

        if near_prompt:
            # Dense attention for tokens near prompts
            for j in range(num_image_tokens):
                row_j = j // grid_size
                col_j = j % grid_size

                # Attend to other tokens near prompts
                for pr, pc in prompt_positions:
                    if abs(row_j - pr) + abs(col_j - pc) <= focus_radius:
                        token_mask[i, j] = True
                        break
        else:
            # Sparse attention for tokens far from prompts
            # Only attend to nearby tokens
            for j in range(num_image_tokens):
                row_j = j // grid_size
                col_j = j % grid_size
                if abs(row_i - row_j) + abs(col_i - col_j) <= 2:
                    token_mask[i, j] = True

    # Convert to block mask
    token_mask_padded = torch.nn.functional.pad(
        token_mask,
        (0, round_to_multiple(total_tokens, block_size) - total_tokens,
         0, round_to_multiple(total_tokens, block_size) - total_tokens)
    )

    block_mask = torch.zeros(nrow, ncol, device=device, dtype=torch.bool)
    for i in range(nrow):
        for j in range(ncol):
            block_mask[i, j] = token_mask_padded[
                i*block_size:(i+1)*block_size,
                j*block_size:(j+1)*block_size
            ].any()

    base_blockmask = block_mask.unsqueeze(0).unsqueeze(0).expand(
        batch_size, num_heads, nrow, ncol
    ).contiguous()

    return base_blockmask


def generate_sam_hybrid_encoder_mask(
    num_image_tokens,
    block_size,
    batch_size,
    num_heads,
    num_dense_heads=4,
    sparsity=0.6,
    device="cuda"
):
    """
    Generate hybrid mask for SAM image encoder only (no prompts yet).
    Follows DuoAttention pattern: some heads dense, some sparse.

    Args:
        num_image_tokens: Number of image tokens
        block_size: Block size for sparse attention
        batch_size: Batch size
        num_heads: Total number of attention heads
        num_dense_heads: Number of heads with dense attention
        sparsity: Sparsity for sparse heads
        device: Device to create tensors on

    Returns:
        base_blockmask: (batch_size, num_sparse_heads, nrow, ncol) boolean mask
        head_mask_type: (num_heads,) tensor indicating mask type per head
    """
    nrow = ncol = round_to_multiple(num_image_tokens, block_size) // block_size

    num_sparse_heads = num_heads - num_dense_heads
    base_mask = torch.zeros(batch_size, num_sparse_heads, nrow, ncol, device=device, dtype=torch.bool)

    grid_size = int(math.sqrt(num_image_tokens))
    density = 1.0 - sparsity

    # Generate sparse masks with spatial bias
    for batch in range(batch_size):
        for head in range(num_sparse_heads):
            for i in range(nrow):
                # For each block row, select blocks with spatial proximity
                num_blocks_to_keep = max(1, int(density * ncol))

                # Simple random selection (could be improved with spatial bias)
                selected_cols = torch.randperm(ncol, device=device)[:num_blocks_to_keep]
                base_mask[batch, head, i, selected_cols] = True

    # Head mask type
    head_mask_type = torch.zeros(num_heads, device=device, dtype=torch.int32)
    head_mask_type[:num_dense_heads] = 0  # Dense
    head_mask_type[num_dense_heads:] = 1  # Sparse

    return base_mask, head_mask_type


def generate_sam_cross_attention_mask(
    num_decoder_tokens,
    num_encoder_tokens,
    block_size,
    batch_size,
    num_heads,
    encoder_sparsity=0.5,
    device="cuda"
):
    """
    Generate mask for SAM decoder cross-attention.
    Decoder tokens (mask embeddings) attend to encoder tokens (image embeddings).

    Args:
        num_decoder_tokens: Number of decoder tokens (mask embeddings)
        num_encoder_tokens: Number of encoder tokens (image embeddings)
        block_size: Block size for sparse attention
        batch_size: Batch size
        num_heads: Number of attention heads
        encoder_sparsity: How sparse the attention to encoder should be
        device: Device to create tensors on

    Returns:
        base_blockmask: (batch_size, num_heads, nrow, ncol) boolean mask
    """
    # Note: Cross-attention is asymmetric (decoder_len x encoder_len)
    nrow = round_to_multiple(num_decoder_tokens, block_size) // block_size
    ncol = round_to_multiple(num_encoder_tokens, block_size) // block_size

    base_mask = torch.zeros(batch_size, num_heads, nrow, ncol, device=device, dtype=torch.bool)

    density = 1.0 - encoder_sparsity

    for batch in range(batch_size):
        for head in range(num_heads):
            # Each decoder token can attend to a subset of encoder tokens
            for i in range(nrow):
                num_blocks_to_keep = max(1, int(density * ncol))
                selected_cols = torch.randperm(ncol, device=device)[:num_blocks_to_keep]
                base_mask[batch, head, i, selected_cols] = True

    return base_mask


# Example usage
if __name__ == "__main__":
    # SAM-B configuration (64x64 image tokens = 4096)
    img_size = 1024  # SAM operates on 1024x1024 images
    patch_size = 16  # ViT-B patch size
    num_image_tokens = (img_size // patch_size) ** 2  # 64x64 = 4096
    num_prompt_tokens = 5  # Example: 2 point prompts + 1 box = 5 tokens
    block_size = 128
    batch_size = 2
    num_heads = 12

    print("=== SAM Block Sparse Attention Mask Examples ===\n")

    # Example 1: Image-to-Prompt mask
    print("1. Image-to-Prompt Mask (sparse image-to-image)")
    mask = generate_sam_image_to_prompt_mask(
        num_image_tokens, num_prompt_tokens, block_size, batch_size, num_heads,
        sparse_image_to_image=True, image_sparsity=0.6
    )
    print(f"   Shape: {mask.shape}")
    print(f"   Total sparsity: {1 - mask.float().mean():.2%}")
    print(f"   Prompt region always attended: True\n")

    # Example 2: Hierarchical mask
    print("2. Hierarchical Mask (different scales per head)")
    mask = generate_sam_hierarchical_mask(
        num_image_tokens, num_prompt_tokens, block_size, batch_size, num_heads
    )
    print(f"   Shape: {mask.shape}")
    for head in [0, num_heads//2, num_heads-1]:
        sparsity = 1 - mask[0, head].float().mean()
        print(f"   Head {head} sparsity: {sparsity:.2%}")
    print()

    # Example 3: Region-focused mask
    print("3. Region-Focused Mask (dense around prompts)")
    # Example: prompts at positions (32, 32) and (48, 48) in 64x64 grid
    prompt_positions = [(32, 32), (48, 48)]
    mask = generate_sam_region_focused_mask(
        num_image_tokens, num_prompt_tokens, prompt_positions,
        block_size, batch_size, num_heads, focus_radius=8
    )
    print(f"   Shape: {mask.shape}")
    print(f"   Sparsity: {1 - mask.float().mean():.2%}")
    print(f"   Dense regions around prompts: {prompt_positions}\n")

    # Example 4: Hybrid encoder (pre-prompt stage)
    print("4. Hybrid Encoder Mask (4 dense + 8 sparse heads)")
    mask, head_types = generate_sam_hybrid_encoder_mask(
        num_image_tokens, block_size, batch_size, num_heads,
        num_dense_heads=4, sparsity=0.6
    )
    print(f"   Sparse mask shape: {mask.shape}")
    print(f"   Head types: {head_types}")
    print(f"   Sparse heads sparsity: {1 - mask.float().mean():.2%}\n")

    # Example 5: Cross-attention mask
    print("5. Cross-Attention Mask (decoder to encoder)")
    num_decoder_tokens = 256  # Mask decoder tokens
    mask = generate_sam_cross_attention_mask(
        num_decoder_tokens, num_image_tokens, block_size, batch_size, num_heads,
        encoder_sparsity=0.5
    )
    print(f"   Shape: {mask.shape}")
    print(f"   Asymmetric: {num_decoder_tokens} decoder → {num_image_tokens} encoder tokens")
    print(f"   Sparsity: {1 - mask.float().mean():.2%}")
