"""
Block Sparse Attention Examples for Vision Models

This package provides ready-to-use mask generation functions for:
- Vision Transformers (ViT)
- Segment Anything Model (SAM)
- Custom vision models

Import mask generators directly:
    from examples.vit_masks import generate_vit_spatial_locality_mask
    from examples.sam_masks import generate_sam_image_to_prompt_mask
"""

from .vit_masks import (
    generate_vit_spatial_locality_mask,
    generate_vit_random_sparse_mask,
    generate_vit_strided_mask,
    generate_vit_hybrid_mask,
    generate_vit_multiscale_mask,
)

from .sam_masks import (
    generate_sam_image_to_prompt_mask,
    generate_sam_hierarchical_mask,
    generate_sam_region_focused_mask,
    generate_sam_hybrid_encoder_mask,
    generate_sam_cross_attention_mask,
)

__all__ = [
    # ViT masks
    'generate_vit_spatial_locality_mask',
    'generate_vit_random_sparse_mask',
    'generate_vit_strided_mask',
    'generate_vit_hybrid_mask',
    'generate_vit_multiscale_mask',
    # SAM masks
    'generate_sam_image_to_prompt_mask',
    'generate_sam_hierarchical_mask',
    'generate_sam_region_focused_mask',
    'generate_sam_hybrid_encoder_mask',
    'generate_sam_cross_attention_mask',
]
