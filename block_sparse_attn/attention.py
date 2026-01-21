"""
Block Sparse Attention - Simplified Python Interface

This module provides a simplified interface for block sparse attention.
Optimized for vision transformer inference with the following assumptions:
- No dropout (always p_dropout=0)
- No causal masking (bidirectional attention)
- No backward pass (inference only)
- No streaming attention

Usage:
    from block_sparse_attn.attention import block_sparse_attn
    from examples.masks import generate_sam_image_to_prompt_mask

    # Generate mask
    mask = generate_sam_image_to_prompt_mask(
        num_image_tokens=4096,
        num_prompt_tokens=5,
        block_size=128,
        batch_size=2,
        num_heads=12
    )

    # Run attention
    output = block_sparse_attn(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        head_mask_type, mask,
        max_seqlen_q, max_seqlen_k
    )
"""

import torch
import block_sparse_attn_cuda  # The compiled C++ extension
from typing import Optional, Tuple


def maybe_contiguous(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Ensure tensor is contiguous in memory."""
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def convert_blockmask_row_reverse(blockmask: torch.Tensor) -> torch.Tensor:
    """
    Convert blockmask from boolean format to the format used by CUDA code.

    Args:
        blockmask: (nrow, ncol) boolean tensor where True means attend to this block

    Returns:
        Converted mask: (nrow, ncol) int32 tensor with row indices in reverse order
    """
    blockmask = blockmask.to(dtype=torch.uint8)
    nonzero_val, nonzero_sorted_rowidx = blockmask.sort(dim=-1, stable=True, descending=False)

    nonzero_idx = nonzero_sorted_rowidx
    nonzero_idx[nonzero_val == 0] = -1
    nonzero_idx = torch.flip(nonzero_idx, dims=[-1])

    return nonzero_idx.contiguous().to(dtype=torch.int32)


def replace_ones_with_count(tensor: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """
    Replace 1s in head_mask_type with sequential counts.

    This is needed to identify which heads use block sparse patterns.

    Args:
        tensor: Head mask type tensor (0=dense, 1=sparse)

    Returns:
        (modified_tensor, count_of_ones)
    """
    ones_mask = tensor == 1
    ones_num = ones_mask.sum()
    count = torch.cumsum(ones_mask, dim=-1).to(tensor.dtype)
    count = count * ones_mask
    tensor = tensor.masked_scatter(ones_mask, count[ones_mask])
    return tensor, ones_num


def prepare_varlen_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seqlen: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Convert standard (batch, seqlen, heads, dim) tensors to varlen format.

    Args:
        q, k, v: (batch_size, seqlen, num_heads, head_dim)
        seqlen: sequence length

    Returns:
        q_unpad, k_unpad, v_unpad: (total_tokens, num_heads, head_dim)
        cu_seqlens: cumulative sequence lengths (batch_size + 1,)
        max_seqlen: maximum sequence length
    """
    batch_size = q.shape[0]
    device = q.device

    # Reshape to (batch * seqlen, heads, dim)
    q_unpad = q.reshape(-1, q.shape[2], q.shape[3])
    k_unpad = k.reshape(-1, k.shape[2], k.shape[3])
    v_unpad = v.reshape(-1, v.shape[2], v.shape[3])

    # Create cumulative sequence lengths: [0, seqlen, 2*seqlen, ..., batch*seqlen]
    cu_seqlens = torch.arange(
        0, (batch_size + 1) * seqlen, step=seqlen,
        dtype=torch.int32, device=device
    )

    return q_unpad, k_unpad, v_unpad, cu_seqlens, seqlen


def block_sparse_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    head_mask_type: torch.Tensor,
    base_blockmask: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    positional: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Block sparse attention (inference only).

    Args:
        q: Query tensor (total_tokens, num_heads, head_dim)
        k: Key tensor (total_tokens, num_heads, head_dim)
        v: Value tensor (total_tokens, num_heads, head_dim)
        cu_seqlens_q: Cumulative sequence lengths for queries (batch_size + 1,)
        cu_seqlens_k: Cumulative sequence lengths for keys (batch_size + 1,)
        head_mask_type: Per-head mask type (num_heads,) - 0=dense, 1=sparse
        base_blockmask: Block sparse mask (batch, num_sparse_heads, nrow, ncol)
        max_seqlen_q: Maximum query sequence length
        max_seqlen_k: Maximum key sequence length
        softmax_scale: Softmax scale factor (default: 1/sqrt(head_dim))
        positional: Optional attention bias (batch, num_heads, seqlen_q, seqlen_k)

    Returns:
        output: Attention output (total_tokens, num_heads, head_dim)

    Notes:
        - All tensors must be on CUDA
        - Supports fp16 and bf16
        - No dropout, no causal masking, no backward pass
        - Optimized for vision transformer inference
    """
    # Ensure tensors are contiguous
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]

    # Set default softmax scale
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    # Handle head mask type
    head_mask_type_modified, blocksparse_head_num = replace_ones_with_count(head_mask_type.clone())

    # Prepare blockmask if provided
    row_blockmask = None
    if base_blockmask is not None:
        assert base_blockmask.shape[1] == blocksparse_head_num, \
            f"Blockmask has {base_blockmask.shape[1]} heads but expected {blocksparse_head_num}"
        row_blockmask = convert_blockmask_row_reverse(base_blockmask)

    # Call C++ extension
    output, softmax_lse = block_sparse_attn_cuda.fwd(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        head_mask_type_modified,
        row_blockmask,
        positional,
        max_seqlen_q, max_seqlen_k,
        softmax_scale
    )

    return output


def block_sparse_attn_simple(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    base_blockmask: Optional[torch.Tensor],
    head_mask_type: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    positional: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Simplified interface that automatically handles varlen conversion.

    Args:
        q, k, v: (batch_size, seqlen, num_heads, head_dim)
        base_blockmask: Block sparse mask (batch, num_heads, nrow, ncol) or None for dense
        head_mask_type: Per-head mask type (num_heads,) - optional, auto-generated if None
        softmax_scale: Softmax scale factor (default: 1/sqrt(head_dim))
        positional: Optional attention bias (batch, num_heads, seqlen_q, seqlen_k)

    Returns:
        output: (batch_size, seqlen, num_heads, head_dim)
    """
    batch_size, seqlen, num_heads, head_dim = q.shape

    # Convert to varlen format
    q_unpad, k_unpad, v_unpad, cu_seqlens, max_seqlen = prepare_varlen_inputs(
        q, k, v, seqlen
    )

    # Auto-generate head_mask_type if not provided
    if head_mask_type is None:
        if base_blockmask is None:
            # All dense
            head_mask_type = torch.zeros(num_heads, dtype=torch.int32, device=q.device)
        else:
            # All sparse
            head_mask_type = torch.ones(num_heads, dtype=torch.int32, device=q.device)

    # Run attention
    output = block_sparse_attn(
        q_unpad, k_unpad, v_unpad,
        cu_seqlens, cu_seqlens,
        head_mask_type,
        base_blockmask,
        max_seqlen, max_seqlen,
        softmax_scale,
        positional
    )

    # Reshape back to (batch, seqlen, heads, dim)
    output = output.reshape(batch_size, seqlen, num_heads, head_dim)

    return output
