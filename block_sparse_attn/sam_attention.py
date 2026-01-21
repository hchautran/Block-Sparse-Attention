"""
SAM Block Sparse Attention compatibility module.

This module mirrors the legacy API expected by examples/docs and forwards
to the current implementation in block_sparse_attn.attention.
"""

from block_sparse_attn.attention import (  # noqa: F401
    block_sparse_attn,
    block_sparse_attn_simple,
    prepare_varlen_inputs,
)

__all__ = [
    "block_sparse_attn",
    "block_sparse_attn_simple",
    "prepare_varlen_inputs",
]
