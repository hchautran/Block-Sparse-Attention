# Block Sparse Attention Examples for ViT and SAM

This directory contains practical examples for using block sparse attention with Vision Transformers (ViT) and Segment Anything Model (SAM).

## Files

### `vit_masks.py`
Mask generation functions for Vision Transformers:

- **`generate_vit_spatial_locality_mask`**: Patches attend primarily to nearby patches
- **`generate_vit_random_sparse_mask`**: Random sparsity with guaranteed CLS token attention
- **`generate_vit_strided_mask`**: Strided attention pattern (every Nth patch)
- **`generate_vit_hybrid_mask`**: Mix of dense and sparse heads (DuoAttention-style)
- **`generate_vit_multiscale_mask`**: Different heads at different spatial scales

### `sam_masks.py`
Mask generation functions for Segment Anything Model:

- **`generate_sam_image_to_prompt_mask`**: Image and prompt tokens with configurable sparsity
- **`generate_sam_hierarchical_mask`**: Multi-scale attention across heads
- **`generate_sam_region_focused_mask`**: Dense attention around prompt regions
- **`generate_sam_hybrid_encoder_mask`**: Hybrid pattern for image encoder
- **`generate_sam_cross_attention_mask`**: Decoder-to-encoder cross-attention

### `usage_example.py`
Complete end-to-end examples showing:
- How to prepare Q, K, V tensors
- How to call `block_sparse_attn_func`
- ViT attention example
- SAM attention example
- Performance comparison

## Quick Start

### Run the Examples

```bash
# Make sure block_sparse_attn is installed
cd /path/to/Block-Sparse-Attention
python setup.py install

# Run the usage examples
cd examples
python usage_example.py

# Test individual mask generators
python vit_masks.py
python sam_masks.py
```

### Basic Usage Pattern

```python
from block_sparse_attn import block_sparse_attn_func
from vit_masks import generate_vit_spatial_locality_mask

# 1. Generate your mask
base_blockmask = generate_vit_spatial_locality_mask(
    img_size=224,
    patch_size=16,
    block_size=128,
    batch_size=4,
    num_heads=12,
    locality_radius=1,
    device="cuda"
)

# 2. Prepare your Q, K, V tensors in varlen format
# Shape: (total_tokens, num_heads, head_dim)
q_unpad = ...
k_unpad = ...
v_unpad = ...
cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32)

# 3. Set head mask types
head_mask_type = torch.ones(num_heads, dtype=torch.int32, device="cuda")

# 4. Run sparse attention
output = block_sparse_attn_func(
    q_unpad, k_unpad, v_unpad,
    cu_seqlens, cu_seqlens,
    head_mask_type,
    streaming_info=None,
    base_blockmask=base_blockmask,
    max_seqlen_q_=seqlen,
    max_seqlen_k_=seqlen,
    p_dropout=0.0,
    is_causal=False,
)
```

## Understanding Sparsity

### How Sparsity Works

The attention matrix is divided into **blocks** (typically 128×128 tokens). The `base_blockmask` is a boolean tensor that specifies which blocks to compute:

```python
base_blockmask[batch, head, row_block, col_block] = True   # Compute this block
base_blockmask[batch, head, row_block, col_block] = False  # Skip this block (zero attention)
```

**Sparsity** = fraction of blocks that are False (skipped)

### Example: 50% Sparsity

```python
# For a 256-token sequence with block_size=128:
# - 2 row blocks, 2 column blocks = 4 total blocks
# - 50% sparsity = skip 2 blocks, compute 2 blocks

base_blockmask = torch.tensor([
    [True,  False],  # Row 0: compute col 0, skip col 1
    [False, True ]   # Row 1: skip col 0, compute col 1
])
```

### Pattern Types

The library supports three pattern types per head (controlled by `head_mask_type`):

1. **Dense (type=0)**: Compute all blocks (no sparsity)
   ```python
   head_mask_type[i] = 0
   # No mask needed, full attention
   ```

2. **Block Sparse (type=1,2,...)**: Use your custom mask
   ```python
   head_mask_type[i] = 1
   # Uses base_blockmask[batch, 0, :, :]  (first sparse head)

   head_mask_type[i] = 2
   # Uses base_blockmask[batch, 1, :, :]  (second sparse head)
   ```

3. **Streaming (type=-1)**: Sink + local window
   ```python
   head_mask_type[i] = -1
   streaming_info = [sink_blocks, local_blocks, ...]
   # Always attend to first 'sink_blocks'
   # Always attend to last 'local_blocks'
   ```

## Vision Model Recommendations

### Vision Transformer (ViT)

**Best patterns:**
- **Spatial Locality** (locality_radius=1-2): Works well since nearby patches are semantically related
- **Hybrid** (4-6 dense heads + rest sparse): Maintains global context while reducing computation
- **Multi-scale**: Different heads capture different receptive fields

**Typical sparsity:** 40-60% for image classification tasks

```python
# Recommended for ViT-B/16
mask, head_types = generate_vit_hybrid_mask(
    img_size=224, patch_size=16, block_size=128,
    batch_size=bs, num_heads=12,
    num_global_heads=4,  # 4 dense, 8 sparse
    sparsity=0.5
)
```

### Segment Anything Model (SAM)

**Best patterns:**
- **Image-to-Prompt**: Always attend image↔prompt, sparse for image-to-image
- **Region-Focused**: Dense attention around prompt locations
- **Hierarchical**: Different scales for different semantic levels

**Typical sparsity:** 50-70% for image encoder, dense for cross-attention

```python
# Recommended for SAM
mask = generate_sam_image_to_prompt_mask(
    num_image_tokens=4096,  # 64x64 patches
    num_prompt_tokens=5,
    block_size=128,
    batch_size=bs, num_heads=12,
    sparse_image_to_image=True,
    image_sparsity=0.6
)
```

## Performance Tips

1. **Block size**: Use 128 (default) for best performance
2. **Head dimension**: 64 and 128 are most optimized
3. **GPU**: Speedup is most significant on A100/H100 GPUs
4. **Data type**: Use fp16 or bf16 for best performance
5. **Sparsity sweet spot**: 40-70% typically gives best speed/accuracy tradeoff

## Customizing Patterns

All mask generation functions are templates. Modify them for your specific use case:

```python
def my_custom_mask(...):
    # Start with a base pattern
    mask = generate_vit_spatial_locality_mask(...)

    # Add your custom logic
    # E.g., always attend to certain special tokens
    mask[:, :, special_token_idx, :] = True
    mask[:, :, :, special_token_idx] = True

    return mask
```

## Troubleshooting

**Q: My attention outputs are wrong**
- Make sure `cu_seqlens` correctly indicates batch boundaries
- Verify mask shape matches number of blocks: `(nrow, ncol)` where `nrow = ceil(seqlen / 128)`

**Q: I get out of memory errors**
- Reduce batch size
- Increase sparsity
- Use fp16 instead of fp32

**Q: No speedup observed**
- Check you're running on GPU
- Verify your mask actually has sparsity (not all True)
- Try higher sparsity (>50%)

**Q: Gradient errors during training**
- Set `deterministic=True` for reproducible gradients
- Ensure all tensors require gradients appropriately

## Citation

If you use this in your research, please cite:

```bibtex
@misc{guo2024blocksparse,
  author       = {Guo, Junxian and Tang, Haotian and Yang, Shang and Zhang, Zhekai and Liu, Zhijian and Han, Song},
  title        = {{Block Sparse Attention}},
  year         = {2024},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/mit-han-lab/Block-Sparse-Attention}}
}
```
