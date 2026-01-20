# Block Sparse Attention

Block Sparse Attention is a high-performance library for efficient sparse attention computation, designed for **Vision Transformers (ViT)** and **Segment Anything Model (SAM)**. By exploiting sparsity in attention patterns, this library significantly reduces computational costs while maintaining model quality.

Originally developed for Large Language Models (LLMs), we provide a focused implementation optimized for vision models. The library supports various sparse patterns including block-sparse attention, streaming attention, and hybrid dense-sparse patterns.

**Key Features for Vision Models:**
- ✨ **Ready-to-use mask generators** for ViT and SAM
- 🚀 **Significant speedup** (1.5-3x) with 40-70% sparsity
- 🎯 **Vision-optimized patterns**: spatial locality, multi-scale attention, prompt-aware masking
- 🔧 **Flexible configuration**: per-head pattern control, hybrid dense-sparse attention
- 📦 **Complete examples** in the `examples/` directory

We release the implementation of Block Sparse Attention, which is initially modified based on [FlashAttention](https://github.com/Dao-AILab/flash-attention) 2.4.2.

![Sparse Patterns](assets/BlockSparseMaskDemo.jpeg)

## News

- [2025/12] We updated the implementation:
  - Support running on Hopper (H100) and Blackwell (B200) GPUs.
  - Clean up Flash Attention supported features.
  - Optimize compilation speed, following Flash Attention 2.8.3.
  - Improve code quality for performance profiling tests.

- [2024/10] We release both fwd pass and bwd pass of Block Sparse Attention.

## Todos
- [2025/12] 
  - [ ] Update code implementation and CUTLASS version to align with FlashAttention 2.8.3.
  - [ ] Refactor internal kernel structure to reduce register pressure.
  - [ ] Optimize block mask storage mechanism to lower peak memory usage.

## Features

We have four patterns supported in Block Sparse Attention:

1. dense attention

   Calculate the full attention matrix.
2. streaming atteniton with token granularity

   Calculate the attention with a fixed number of sink tokens and local tokens. You can refer to [StreamingLLM](https://arxiv.org/abs/2309.17453) for more details.
3. streaming attention with block granularity, block_size = 128

   Calculate the attention with a fixed number of sink blocks and local blocks.
4. blocksparse attention, block_size = 128

   Take in a block mask and calculate the attention with the block mask.

**Importantly, we support assigning different patterns for different heads.**

You can use `head_mask_type` to specify the pattern for each head. This is a list of quiry head number of integers.

For one head, `mask_type = 0` means dense attention, `mask_type = -1` means streaming attention (either block streaming or exact streaming), and `mask_type = 1` means blocksparse attention, the head will use `basemask[mask_type - 1]` as its attention mask.

For example, if you have 8 heads and

```python
    head_mask_type = [1, 1, 0, 0, 0, -1, 0, -1]
```

This means head0, head1 use blocksparse mask, head2 to head4 and head 6 use dense mask, and head 5 and head 7 use streaming mask.

The interface is:

```python
from block_sparse_attn import block_sparse_attn_func
block_sparse_attn_func(
    q_unpad, k_unpad, v_unpad,
    cu_seqlens_q, cu_seqlens_k,
    head_mask_type,
    streaming_info,
    base_blockmask,
    max_seqlen_q_, max_seqlen_k_,
    p_dropout,
    deterministic=False,
    softmax_scale=None,
    is_causal=False,
    exact_streaming=False,
    return_attn_probs=False,
)
```

```python
from block_sparse_attn import block_streaming_attn_func
block_streaming_attn_func(
    q_unpad, k_unpad, v_unpad,
    cu_seqlens_q, cu_seqlens_k,
    head_mask_type,
    streaming_info,
    max_seqlen_q, max_seqlen_k,
    p_dropout,
    deterministic=False,
    softmax_scale=None,
    is_causal=True,
    return_attn_probs=False,
)
```

```python
from block_sparse_attn import token_streaming_attn_func
# bwd pass is not yet supported
token_streaming_attn_func(
    q_unpad, k_unpad, v_unpad,
    cu_seqlens_q, cu_seqlens_k,
    head_mask_type,
    streaming_info,
    max_seqlen_q, max_seqlen_k,
    deterministic=False,
    softmax_scale=None,
    return_attn_probs=False,
)
```

## Performance

### Block Sparse Speedup

<div align=center><img src="assets/BlocksparseSpeedUp.jpeg"></div>

<div align=center><img src="assets/BlocksparseSpeedUpFwdBwd.jpeg"></div>

The figures above illustrate the speedup gained by using Block Sparse Attention in comparison to dense FlashAttention2 2.4.2. This speedup was measured on an A100 GPU, with configurations including a head dimension of 128 and 32 attention heads.

### Dense & Streaming Hybrid Speedup

[Duo Attention](https://github.com/mit-han-lab/duo-attention) introduces a hybrid mask scenario, where half of the attention heads utilize a dense mask and the other half employ a streaming mask. This pattern is also proved to be an accurate approach for LLMs inference.

<div align=center><img src="assets/StreamingHybridSpeedUpRatio.jpeg"></div>

The graph above demonstrates the performance of our kernel for this specified workload. For token-level streaming masks, we allocate 64 sink tokens and 256 local tokens. For block-level streaming masks, we allocate 1 sink block and 3 local blocks, with each block consisting of 128 tokens. Speedup results were measured on an A100 GPU, using dense FlashAttention2 as the baseline, with a head dimension of 128, 32 attention heads, and a batch size of 1.

## Installation

Requirements:

- CUDA 11.6 and above.
- PyTorch 1.12 and above.
- Linux.

```sh
pip install packaging
pip install ninja
python setup.py install
```

Block Sparse Interface: `block_sparse_attn/block_sparse_attn_interface.py`

Block Sparse Attention currently supports:

1. Datatype fp16 and bf16 (bf16 requires Ampere, Ada, or Hopper GPUs).
2. Head dimension 32, 64, 128.

## Quick Start for Vision Models

### Vision Transformer (ViT)

```python
from block_sparse_attn import block_sparse_attn_func
from examples.vit_masks import generate_vit_spatial_locality_mask

# Generate spatial locality mask for ViT-B/16
base_blockmask = generate_vit_spatial_locality_mask(
    img_size=224,          # Image size
    patch_size=16,         # ViT-B/16 patch size
    block_size=128,        # Block size for sparse attention
    batch_size=4,
    num_heads=12,
    locality_radius=1,     # Attend to immediate neighbors
    include_cls_token=True,
    device="cuda"
)

# Prepare Q, K, V tensors (in varlen format)
# See examples/usage_example.py for complete code
output = block_sparse_attn_func(
    q_unpad, k_unpad, v_unpad,
    cu_seqlens_q, cu_seqlens_k,
    head_mask_type,
    streaming_info=None,
    base_blockmask=base_blockmask,
    max_seqlen_q_=num_tokens,
    max_seqlen_k_=num_tokens,
    p_dropout=0.0,
    is_causal=False,
)
```

### Segment Anything Model (SAM)

```python
from examples.sam_masks import generate_sam_image_to_prompt_mask

# Generate mask where image tokens attend to prompts
base_blockmask = generate_sam_image_to_prompt_mask(
    num_image_tokens=4096,      # 64x64 patches for SAM-B
    num_prompt_tokens=5,        # Point and box prompts
    block_size=128,
    batch_size=2,
    num_heads=12,
    sparse_image_to_image=True, # Sparse for image-to-image
    image_sparsity=0.6,         # 60% sparsity
    device="cuda"
)

# Use with block_sparse_attn_func (same as above)
```

**See [`examples/`](examples/) directory for complete working examples!**

### Tests

To run the correctness tests:
```sh
pip install pytest
```

- For fwd only

  ```sh
  cd ./block_sparse_tests/fwd/test_correctness
  pytest full_test.py
  ```
- For fwd and bwd

  ```sh
  cd ./block_sparse_tests/fwd_bwd/test_correctness
  pytest full_test.py
  ```

To run the performance tests:

- For fwd only

  ```sh
  cd ./block_sparse_tests/fwd/test_performance/
  python token_streaming.py
  python blocksparse.py
  ```
- For fwd and bwd

  ```sh
  cd ./block_sparse_tests/fwd_bwd/test_performance/
  python block_streaming.py
  python blocksparse.py
  ```

## Team

| | |
| --- | --- |
[Junxian Guo](https://github.com/JerryGJX): SJTU, MIT|  [Haotian Tang](http://kentang.net/): MIT
[Shang Yang](https://ys-2020.github.io/): MIT        |  [Zhekai Zhang](https://hanlab.mit.edu/team/zhekai-zhang): MIT
[Zhijian Liu](https://zhijianliu.com/): Nvidia, MIT  |  [Song Han](https://hanlab.mit.edu/songhan): Nvidia, MIT



## Acknowledgement

- [FlashAttention](https://github.com/Dao-AILab/flash-attention): the codebase we built upon. Thanks for their wonderful work. The design of block sparse attention in FlashAttention v1.0 is very inspiring.
- [FlashAttention](https://arxiv.org/abs/2205.14135), [FlashAttention-2](https://arxiv.org/abs/2307.08691), [Big Bird](https://arxiv.org/abs/2007.14062), [ETC](https://arxiv.org/abs/2004.08483): get the idea of block sparse attention and how it can be implemented.
- [StreamingLLM](https://arxiv.org/abs/2309.17453): get the idea of streaming attention.
- [Duo Attention](https://github.com/mit-han-lab/duo-attention), [MInference 1.0](https://arxiv.org/abs/2407.02490): get the idea of hybrid masks.

## Related Projects

- [DuoAttention](https://arxiv.org/abs/2410.10819): Efficient Long-Context LLM Inference with Retrieval and Streaming Heads
- [LServe](https://arxiv.org/abs/2502.14866): Efficient Long-sequence LLM Serving with Unified Sparse Attention
- [XAttention](https://arxiv.org/abs/2508.11131): Block sparse attention with antidiagonal scoring



## Citation

```
@misc{guo2024blocksparse,
  author       = {Guo, Junxian and Tang, Haotian and Yang, Shang and Zhang, Zhekai and Liu, Zhijian and Han, Song},
  title        = {{Block Sparse Attention}},
  year         = {2024},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/mit-han-lab/Block-Sparse-Attention}}
}

```