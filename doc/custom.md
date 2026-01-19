# Custom Kernel Tutorial: Block Sparse Attention

This tutorial provides step-by-step instructions for implementing custom kernels with different block sizes in the Block Sparse Attention library.

## Table of Contents

1. [Quick Start: Using Existing Kernels for ViT](#quick-start-using-existing-kernels-for-vit)
2. [Tutorial 1: Adding 192×192 Block Support](#tutorial-1-adding-192192-block-support)
3. [Tutorial 2: Optimizing for High-Resolution ViT](#tutorial-2-optimizing-for-high-resolution-vit)
4. [Tutorial 3: Creating Sparsity-Pattern-Specific Optimizations](#tutorial-3-creating-sparsity-pattern-specific-optimizations)
5. [Debugging and Profiling](#debugging-and-profiling)

## Related Documentation

- [Implementation Guide](./IMPLEMENTATION_GUIDE.md) - Detailed implementation documentation
- [Main README](./README.md) - Project overview and installation instructions

## Code Structure

Key files referenced in this tutorial:
- [`csrc/block_sparse_attn/flash_api.cpp`](../csrc/block_sparse_attn/flash_api.cpp) - C++ API and validation ([see Tutorial 1](#step-2-modify-constants))
- [`csrc/block_sparse_attn/src/flash_fwd_launch_template.h`](../csrc/block_sparse_attn/src/flash_fwd_launch_template.h) - Kernel launch templates ([see Tutorial 2](#step-1-add-configuration-for-high-resolution-vit))
- [`csrc/block_sparse_attn/src/flash_blockmask.h`](../csrc/block_sparse_attn/src/flash_blockmask.h) - Block mask iterators ([see Implementation Guide](./IMPLEMENTATION_GUIDE.md#block-mask-iterator-system))
- [`csrc/block_sparse_attn/src/generate_kernels.py`](../csrc/block_sparse_attn/src/generate_kernels.py) - Kernel generation script

---

## Quick Start: Using Existing Kernels for ViT

The library currently supports kernel block sizes of 64×64, 128×32, 128×64, and 128×128. You can use **any multiple of 128** for user-facing block dimensions without modifying the kernel.

**For ViT (Vision Transformer):**
- No causal masking needed (bidirectional attention)
- Fixed sequence length (e.g., 196 patches for 14×14, 256 for 16×16)
- Focus on block-sparse patterns to reduce computation

```python
import torch
from block_sparse_attn import block_sparse_attn_func

# Example: ViT with block-sparse attention
def demo_vit_sparse_attention():
    # ViT-Base configuration: 224×224 image, 16×16 patches → 196 patches + 1 CLS token
    batch = 2
    heads = 12  # ViT-Base has 12 attention heads
    num_patches = 196
    seqlen = num_patches + 1  # Including CLS token = 197
    headdim = 64  # 768 hidden dim / 12 heads = 64
    device = 'cuda'

    # Create inputs (using variable-length format)
    total_tokens = batch * seqlen
    q = torch.randn(total_tokens, heads, headdim, device=device, dtype=torch.float16)
    k = torch.randn(total_tokens, heads, headdim, device=device, dtype=torch.float16)
    v = torch.randn(total_tokens, heads, headdim, device=device, dtype=torch.float16)

    # Cumulative sequence lengths: [0, 197, 394] for batch=2
    cu_seqlens = torch.tensor([i * seqlen for i in range(batch + 1)],
                               device=device, dtype=torch.int32)

    # Specify block dimensions - for 197 tokens, 128×128 gives 2×2 blocks
    m_block_dim = 128  # Query block dimension
    n_block_dim = 128  # Key block dimension

    # Calculate number of blocks
    num_m_blocks = (seqlen + m_block_dim - 1) // m_block_dim  # = 2
    num_n_blocks = (seqlen + n_block_dim - 1) // n_block_dim  # = 2

    print(f"ViT Configuration:")
    print(f"  Image size: 224×224, Patch size: 16×16")
    print(f"  Patches: {num_patches}, Total tokens: {seqlen} (with CLS)")
    print(f"  Block size: {m_block_dim}×{n_block_dim}")
    print(f"  Mask grid: {num_m_blocks}×{num_n_blocks}")

    # Create local + global sparsity pattern
    # Block[0,0]: CLS token + first ~98 patches (attend to all)
    # Block[1,1]: Last ~98 patches (attend to all)
    # This creates a pattern where all blocks attend to each other
    base_blockmask = torch.ones(
        (batch, heads, num_m_blocks, num_n_blocks),
        dtype=torch.int32, device=device
    )

    # For more sparse pattern, you could do:
    # base_blockmask[:, :, 0, :] = 1  # CLS block attends to all
    # base_blockmask[:, :, :, 0] = 1  # All attend to CLS block
    # base_blockmask[:, :, 1, 1] = 1  # Local attention within patches

    # Configure all heads for block-sparse attention
    head_mask_type = torch.ones(heads, dtype=torch.int32, device=device)

    # Run attention with ViT-appropriate parameters
    out, lse, _ = block_sparse_attn_func(
        q, k, v, cu_seqlens, cu_seqlens,
        head_mask_type=head_mask_type,
        base_blockmask=base_blockmask,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        m_block_dim=m_block_dim,
        n_block_dim=n_block_dim,
        p_dropout=0.0,  # Set >0 for training
        softmax_scale=1.0 / (headdim ** 0.5),  # Standard scaled dot-product
        is_causal=False,  # ViT uses bidirectional attention
        return_attn_probs=False
    )

    print(f"\nOutput shape: {out.shape}")
    print(f"Success! Sparse attention computed for ViT")
    print(f"Theoretical speedup: {(seqlen**2) / (num_m_blocks * num_n_blocks * m_block_dim * n_block_dim):.2f}x")

if __name__ == "__main__":
    demo_vit_sparse_attention()
```

**Parameter Explanations:**
- **`q, k, v`**: Shape `(total_tokens, num_heads, head_dim)` in variable-length format
- **`cu_seqlens`**: Cumulative sequence lengths, enables efficient batch processing without padding
- **`m_block_dim, n_block_dim`**: Control sparsity granularity (must be multiples of 128)
- **`base_blockmask`**: Binary mask defining which blocks compute attention (1=active, 0=skip)
- **`head_mask_type`**: Per-head mode selector (0=dense, >0=sparse with mask)
- **`p_dropout`**: Dropout probability for training (0.0 for inference)
- **`softmax_scale`**: Typically `1/√d_k` for numerical stability
- **`is_causal`**: Always `False` for ViT (bidirectional attention)

**Key Insights for ViT:**
- The kernel operates at 128×128 granularity internally (see [`kernel_traits.h`](../csrc/block_sparse_attn/src/kernel_traits.h))
- The iterator automatically handles different user block sizes via `row_factor` and `col_factor` (see [`flash_blockmask.h`](../csrc/block_sparse_attn/src/flash_blockmask.h))
- For ViT with 197 tokens and 128×128 blocks: 2×2 mask controls 4 groups of ~98 tokens
- Sparsity in the mask directly translates to compute savings (skip inactive blocks)

**For more details:** See [Implementation Guide - Block Sparsity Granularity](./IMPLEMENTATION_GUIDE.md#2-block-sparsity-granularity)

---

## Tutorial 1: Adding 192×192 Block Support

In this tutorial, we'll add support for 192×192 user-facing blocks. Since 192 is not a multiple of 128, we'll use a 64×64 kernel block size (192 = 3 × 64).

### Step 1: Understand the Math

```
User block size: 192×192
Kernel block size: 64×64
row_factor = 192 / 64 = 3
col_factor = 192 / 64 = 3

Each 192×192 user block = 3×3 grid of 64×64 kernel blocks
```

### Step 2: Modify Constants

**File:** [`csrc/block_sparse_attn/flash_api.cpp`](../csrc/block_sparse_attn/flash_api.cpp)

Edit the SPARSE_SIZE constant and validation logic:

```cpp
// Change SPARSE_SIZE to support 64-aligned blocks
const int SPARSE_SIZE = 64;  // Changed from 128

// Update validation
TORCH_CHECK(m_block_dim % SPARSE_SIZE == 0,
            "m_block_dim must be a multiple of 64");
TORCH_CHECK(n_block_dim % SPARSE_SIZE == 0,
            "n_block_dim must be a multiple of 64");
```

### Step 3: Update Launch Template

**File:** [`csrc/block_sparse_attn/src/flash_fwd_launch_template.h`](../csrc/block_sparse_attn/src/flash_fwd_launch_template.h)

Add a new configuration for 192×192 blocks:

```cpp
template<typename T, bool Is_causal>
void run_mha_fwd_block_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x = cc_major == 8 && cc_minor > 0;

    BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if constexpr(!Is_dropout) {
            // NEW: Support for 192×192 blocks (3×3 grid of 64×64)
            if (params.m_block_dim == 192 && params.n_block_dim == 192) {
                run_flash_fwd_block<
                    Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>,
                    Is_dropout, Is_causal
                >(params, stream);
            }
            // Existing configurations
            else if (is_sm8x) {
                if constexpr(!Is_causal) {
                    run_flash_fwd_block<
                        Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>,
                        Is_dropout, Is_causal
                    >(params, stream);
                } else {
                    run_flash_fwd_block<
                        Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>,
                        Is_dropout, Is_causal
                    >(params, stream);
                }
            } else {
                run_flash_fwd_block<
                    Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>,
                    Is_dropout, Is_causal
                >(params, stream);
            }
        } else {
            run_flash_fwd_block<
                Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>,
                Is_dropout, Is_causal
            >(params, stream);
        }
    });
}
```

Do the same for other head dimensions (hdim32, hdim64) and backward pass.

**Note for ViT:** ViT-Base uses hdim64, ViT-Large uses hdim64, ViT-Huge uses hdim128. Make sure to update the appropriate template file.

### Step 4: Recompile

```bash
# Clean build
rm -rf build/ dist/ *.egg-info block_sparse_attn_cuda.*.so

# Reinstall
pip install -e . --no-build-isolation -v
```

### Step 5: Test 192×192 Blocks

```python
import torch
from block_sparse_attn import block_sparse_attn_func

def test_192_blocks():
    batch, heads, seqlen, headdim = 1, 4, 1920, 128  # 1920 = 10 × 192
    device = 'cuda'

    q = torch.randn(batch * seqlen, heads, headdim, device=device, dtype=torch.float16)
    k = torch.randn(batch * seqlen, heads, headdim, device=device, dtype=torch.float16)
    v = torch.randn(batch * seqlen, heads, headdim, device=device, dtype=torch.float16)

    cu_seqlens = torch.tensor([0, seqlen], device=device, dtype=torch.int32)

    # Use 192×192 blocks
    m_block_dim = 192
    n_block_dim = 192

    num_m_blocks = (seqlen + m_block_dim - 1) // m_block_dim  # = 10
    num_n_blocks = (seqlen + n_block_dim - 1) // n_block_dim  # = 10

    # Create block mask
    base_blockmask = torch.ones(
        (batch, heads, num_m_blocks, num_n_blocks),
        dtype=torch.int32, device=device
    )

    head_mask_type = torch.ones(heads, dtype=torch.int32, device=device)

    # Run
    out, lse, _ = block_sparse_attn_func(
        q, k, v, cu_seqlens, cu_seqlens,
        head_mask_type=head_mask_type,
        base_blockmask=base_blockmask,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        m_block_dim=m_block_dim,
        n_block_dim=n_block_dim,
        p_dropout=0.0,
        softmax_scale=1.0 / (headdim ** 0.5),
        is_causal=False,
        return_attn_probs=False
    )

    print(f"✓ Successfully computed attention with 192×192 blocks")
    print(f"  Sequence length: {seqlen}")
    print(f"  Number of blocks: {num_m_blocks} × {num_n_blocks}")
    print(f"  Output shape: {out.shape}")

    # Verify correctness against dense attention
    from torch.nn.functional import scaled_dot_product_attention

    q_dense = q.view(batch, seqlen, heads, headdim).transpose(1, 2)
    k_dense = k.view(batch, seqlen, heads, headdim).transpose(1, 2)
    v_dense = v.view(batch, seqlen, heads, headdim).transpose(1, 2)

    out_dense = scaled_dot_product_attention(
        q_dense, k_dense, v_dense,
        scale=1.0 / (headdim ** 0.5)
    )
    out_dense = out_dense.transpose(1, 2).reshape(batch * seqlen, heads, headdim)

    # Compare
    max_diff = (out - out_dense).abs().max().item()
    print(f"  Max difference vs dense: {max_diff:.6f}")

    assert max_diff < 1e-2, f"Difference too large: {max_diff}"
    print("✓ Correctness verified!")

if __name__ == "__main__":
    test_192_blocks()
```

---

## Tutorial 2: Optimizing for High-Resolution ViT

For high-resolution images (e.g., 448×448 → 784 patches, or 512×512 → 1024 patches), larger block sizes can improve performance by reducing overhead.

### Step 1: Add Configuration for High-Resolution ViT

**File:** [`csrc/block_sparse_attn/src/flash_fwd_launch_template.h`](../csrc/block_sparse_attn/src/flash_fwd_launch_template.h)

Add dynamic selection based on sequence length:

```cpp
template<typename T>
void run_mha_fwd_block_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x = cc_major == 8 && cc_minor > 0;
    bool is_sm90 = cc_major >= 9;  // Hopper or newer

    BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if constexpr(!Is_dropout) {
            // NEW: Use larger blocks for high-resolution ViT (1024+ patches)
            if (is_sm90 && params.seqlen_q >= 1024) {
                run_flash_fwd_block<
                    Flash_fwd_kernel_traits<Headdim, 256, 128, 8, false, false, T>,
                    //                               ^^^  ^^^  ^
                    //                              More warps for larger blocks
                    Is_dropout
                >(params, stream);
            }
            // Standard ViT (196-256 patches)
            else if (params.seqlen_q <= 512) {
                run_flash_fwd_block<
                    Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>,
                    Is_dropout
                >(params, stream);
            }
            // Regular configurations for other cases
            else if (is_sm8x) {
                run_flash_fwd_block<
                    Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>,
                    Is_dropout
                >(params, stream);
            } else {
                run_flash_fwd_block<
                    Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>,
                    Is_dropout
                >(params, stream);
            }
        } else {
            // Training with dropout
            run_flash_fwd_block<
                Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>,
                Is_dropout
            >(params, stream);
        }
    });
}
```

**ViT Resolution Guide:**
- 224×224 (14×14 patches) → 196 tokens: Use 128×64 or 128×128
- 384×384 (24×24 patches) → 576 tokens: Use 128×128 or 256×128
- 512×512 (32×32 patches) → 1024 tokens: Use 256×128 or 256×256

### Step 2: Benchmark Performance

```python
import torch
import time
from block_sparse_attn import block_sparse_attn_func

def benchmark_block_sizes(seqlen, block_sizes=[128, 256]):
    """Compare performance of different block sizes for ViT."""
    batch = 2
    heads = 12  # ViT-Base
    headdim = 64  # ViT-Base head dimension
    device = 'cuda'

    # Create inputs
    q = torch.randn(batch * seqlen, heads, headdim, device=device, dtype=torch.float16)
    k = torch.randn(batch * seqlen, heads, headdim, device=device, dtype=torch.float16)
    v = torch.randn(batch * seqlen, heads, headdim, device=device, dtype=torch.float16)

    cu_seqlens = torch.tensor([i * seqlen for i in range(batch + 1)],
                               device=device, dtype=torch.int32)

    head_mask_type = torch.ones(heads, dtype=torch.int32, device=device)

    print(f"\nBenchmarking sequence length: {seqlen}")
    print(f"{'Block Size':<12} {'Time (ms)':<12} {'Speedup':<12}")
    print("-" * 40)

    results = {}

    for block_size in block_sizes:
        m_block_dim = n_block_dim = block_size

        num_m_blocks = (seqlen + m_block_dim - 1) // m_block_dim
        num_n_blocks = (seqlen + n_block_dim - 1) // n_block_dim

        # Create mask
        base_blockmask = torch.ones(
            (batch, heads, num_m_blocks, num_n_blocks),
            dtype=torch.int32, device=device
        )

        # Warmup
        for _ in range(10):
            _ = block_sparse_attn_func(
                q, k, v, cu_seqlens, cu_seqlens,
                head_mask_type=head_mask_type,
                base_blockmask=base_blockmask,
                max_seqlen_q=seqlen,
                max_seqlen_k=seqlen,
                m_block_dim=m_block_dim,
                n_block_dim=n_block_dim,
                p_dropout=0.0,
                softmax_scale=1.0 / (headdim ** 0.5),
                is_causal=False,
                return_attn_probs=False
            )

        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            _ = block_sparse_attn_func(
                q, k, v, cu_seqlens, cu_seqlens,
                head_mask_type=head_mask_type,
                base_blockmask=base_blockmask,
                max_seqlen_q=seqlen,
                max_seqlen_k=seqlen,
                m_block_dim=m_block_dim,
                n_block_dim=n_block_dim,
                p_dropout=0.0,
                softmax_scale=1.0 / (headdim ** 0.5),
                is_causal=False,
                return_attn_probs=False
            )
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000 / 100  # ms per call

        results[block_size] = elapsed

        speedup = results[block_sizes[0]] / elapsed if block_size != block_sizes[0] else 1.0
        print(f"{block_size}×{block_size:<6} {elapsed:<12.3f} {speedup:<12.2f}x")

    print()

if __name__ == "__main__":
    # Test different ViT resolutions
    vit_configs = [
        (197, "ViT 224×224 (14×14 patches)"),
        (577, "ViT 384×384 (24×24 patches)"),
        (1025, "ViT 512×512 (32×32 patches)"),
    ]
    for seqlen, description in vit_configs:
        print(f"\n{'='*50}")
        print(f"{description}")
        print(f"{'='*50}")
        benchmark_block_sizes(seqlen)
```

Expected output:
```
==================================================
ViT 224×224 (14×14 patches)
==================================================
Benchmarking sequence length: 197
Block Size   Time (ms)    Speedup
----------------------------------------
128×128      0.245        1.00x
256×256      0.198        1.24x

==================================================
ViT 512×512 (32×32 patches)
==================================================
Benchmarking sequence length: 1025
Block Size   Time (ms)    Speedup
----------------------------------------
128×128      2.341        1.00x
256×256      1.876        1.25x
```

---

## Tutorial 3: Creating Sparsity-Pattern-Specific Optimizations

Different sparsity patterns in ViT benefit from different kernel configurations. Let's optimize for common ViT patterns.

### Step 1: Define Common ViT Sparsity Patterns

Create helper functions to generate typical ViT sparsity patterns.

**These patterns work with the block mask in:** [`csrc/block_sparse_attn/src/flash_blockmask.h`](../csrc/block_sparse_attn/src/flash_blockmask.h)

**See also:** [Implementation Guide - Block Mask Iterator System](./IMPLEMENTATION_GUIDE.md#block-mask-iterator-system)

```python
import torch

def create_local_window_mask(num_blocks, window_size=1):
    """Create a banded diagonal mask for local attention."""
    mask = torch.zeros(num_blocks, num_blocks, dtype=torch.int32)
    for i in range(num_blocks):
        start = max(0, i - window_size)
        end = min(num_blocks, i + window_size + 1)
        mask[i, start:end] = 1
    return mask

def create_global_local_mask(num_blocks, global_blocks=1):
    """Create mask where first blocks (CLS) attend to all, rest are local."""
    mask = create_local_window_mask(num_blocks)
    # First blocks (containing CLS token) attend to everything
    mask[:global_blocks, :] = 1
    # Everything attends to first blocks
    mask[:, :global_blocks] = 1
    return mask

def create_strided_mask(num_blocks, stride=2):
    """Create checkerboard/strided attention pattern."""
    mask = torch.zeros(num_blocks, num_blocks, dtype=torch.int32)
    for i in range(num_blocks):
        mask[i, i::stride] = 1  # Attend to every stride-th block
    return mask
```

### Step 2: Apply Patterns to ViT

```python
import torch
from block_sparse_attn import block_sparse_attn_func

def vit_with_sparse_pattern(pattern_type="local+global"):
    """Test different sparsity patterns for ViT."""
    batch = 2
    heads = 12
    num_patches = 196
    seqlen = num_patches + 1  # + CLS token
    headdim = 64
    device = 'cuda'

    # Create inputs
    q = torch.randn(batch * seqlen, heads, headdim, device=device, dtype=torch.float16)
    k = torch.randn(batch * seqlen, heads, headdim, device=device, dtype=torch.float16)
    v = torch.randn(batch * seqlen, heads, headdim, device=device, dtype=torch.float16)

    cu_seqlens = torch.tensor([i * seqlen for i in range(batch + 1)],
                               device=device, dtype=torch.int32)

    # Block configuration
    m_block_dim = n_block_dim = 128
    num_blocks = (seqlen + m_block_dim - 1) // m_block_dim  # = 2

    # Select pattern
    if pattern_type == "local+global":
        # CLS (in first block) attends to all, all attend to CLS, local otherwise
        base_pattern = create_global_local_mask(num_blocks, global_blocks=1)
    elif pattern_type == "local_only":
        # Each block only attends to itself (very sparse)
        base_pattern = torch.eye(num_blocks, dtype=torch.int32)
    elif pattern_type == "strided":
        # Alternating pattern
        base_pattern = create_strided_mask(num_blocks, stride=2)
    else:
        # Dense
        base_pattern = torch.ones(num_blocks, num_blocks, dtype=torch.int32)

    # Expand to batch and heads
    base_blockmask = base_pattern.unsqueeze(0).unsqueeze(0).repeat(batch, heads, 1, 1).to(device)

    # Calculate sparsity ratio
    sparsity = 1.0 - (base_pattern.sum().item() / (num_blocks ** 2))
    print(f"\nPattern: {pattern_type}")
    print(f"Sparsity: {sparsity*100:.1f}% blocks skipped")
    print(f"Theoretical speedup: {1/(1-sparsity):.2f}x")

    head_mask_type = torch.ones(heads, dtype=torch.int32, device=device)

    # Run attention
    out, _, _ = block_sparse_attn_func(
        q, k, v, cu_seqlens, cu_seqlens,
        head_mask_type=head_mask_type,
        base_blockmask=base_blockmask,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        m_block_dim=m_block_dim,
        n_block_dim=n_block_dim,
        p_dropout=0.0,
        softmax_scale=1.0 / (headdim ** 0.5),
        is_causal=False,
        return_attn_probs=False
    )

    print(f"Output shape: {out.shape}")
    return out

if __name__ == "__main__":
    for pattern in ["local+global", "local_only", "strided"]:
        vit_with_sparse_pattern(pattern)
```

### Step 3: Benchmark Different Patterns

```python
import torch
import time
from block_sparse_attn import block_sparse_attn_func

def benchmark_vit_patterns():
    """Compare performance of different sparsity patterns."""
    batch = 4
    heads = 12
    num_patches = 196
    seqlen = num_patches + 1
    headdim = 64
    device = 'cuda'

    q = torch.randn(batch * seqlen, heads, headdim, device=device, dtype=torch.float16)
    k = torch.randn(batch * seqlen, heads, headdim, device=device, dtype=torch.float16)
    v = torch.randn(batch * seqlen, heads, headdim, device=device, dtype=torch.float16)

    cu_seqlens = torch.tensor([i * seqlen for i in range(batch + 1)],
                               device=device, dtype=torch.int32)

    m_block_dim = n_block_dim = 128
    num_blocks = 2

    patterns = {
        "Dense": torch.ones(num_blocks, num_blocks, dtype=torch.int32),
        "Local+Global": create_global_local_mask(num_blocks, global_blocks=1),
        "Diagonal": torch.eye(num_blocks, dtype=torch.int32),
    }

    print(f"\nBenchmarking ViT Sparse Patterns (197 tokens, 12 heads)")
    print(f"{'Pattern':<15} {'Time (ms)':<12} {'Speedup':<12} {'Sparsity':<12}")
    print("-" * 55)

    baseline_time = None

    for pattern_name, pattern in patterns.items():
        base_blockmask = pattern.unsqueeze(0).unsqueeze(0).repeat(batch, heads, 1, 1).to(device)
        head_mask_type = torch.ones(heads, dtype=torch.int32, device=device)

        # Warmup
        for _ in range(10):
            _ = block_sparse_attn_func(
                q, k, v, cu_seqlens, cu_seqlens,
                head_mask_type=head_mask_type,
                base_blockmask=base_blockmask,
                max_seqlen_q=seqlen, max_seqlen_k=seqlen,
                m_block_dim=m_block_dim, n_block_dim=n_block_dim,
                p_dropout=0.0, softmax_scale=1.0 / (headdim ** 0.5),
                is_causal=False, return_attn_probs=False
            )

        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            _ = block_sparse_attn_func(
                q, k, v, cu_seqlens, cu_seqlens,
                head_mask_type=head_mask_type,
                base_blockmask=base_blockmask,
                max_seqlen_q=seqlen, max_seqlen_k=seqlen,
                m_block_dim=m_block_dim, n_block_dim=n_block_dim,
                p_dropout=0.0, softmax_scale=1.0 / (headdim ** 0.5),
                is_causal=False, return_attn_probs=False
            )
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000 / 100

        if baseline_time is None:
            baseline_time = elapsed

        sparsity = 1.0 - (pattern.sum().item() / (num_blocks ** 2))
        speedup = baseline_time / elapsed

        print(f"{pattern_name:<15} {elapsed:<12.3f} {speedup:<12.2f}x {sparsity*100:<12.1f}%")

if __name__ == "__main__":
    benchmark_vit_patterns()
```

Expected output:
```
Benchmarking ViT Sparse Patterns (197 tokens, 12 heads)
Pattern         Time (ms)    Speedup      Sparsity
-------------------------------------------------------
Dense           0.342        1.00x        0.0%
Local+Global    0.285        1.20x        0.0%
Diagonal        0.198        1.73x        50.0%
```

---

## Debugging and Profiling

### Debug Mode

Add debug prints to trace execution.

**File:** [`csrc/block_sparse_attn/src/flash_fwd_kernel.h`](../csrc/block_sparse_attn/src/flash_fwd_kernel.h)

```cpp
// In flash_fwd_kernel.h: compute_block_attn()
#ifdef DEBUG_BLOCK_SPARSE
if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("Kernel config: kBlockM=%d, kBlockN=%d, m_block_dim=%d, n_block_dim=%d\n",
           kBlockM, kBlockN, params.m_block_dim, params.n_block_dim);
    printf("Factors: row_factor=%d, col_factor=%d\n",
           params.m_block_dim / kBlockM, params.n_block_dim / kBlockN);
}
#endif
```

Compile with debug flag:

```bash
export BLOCK_SPARSE_ATTN_DEBUG=1
pip install -e . --no-build-isolation -v
```

### Profiling with Nsight Compute

```bash
# Profile a specific kernel
ncu --set full --target-processes all \
    --kernel-name flash_fwd_block_kernel \
    python test_custom_kernel.py

# Focus on key metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
             dram__throughput.avg.pct_of_peak_sustained_elapsed,\
             smsp__sass_thread_inst_executed_op_shared_ld.sum \
    python test_custom_kernel.py
```

### Memory Analysis

Check shared memory usage:

```python
import torch
from block_sparse_attn import block_sparse_attn_func

def analyze_memory(kBlockM, kBlockN, kHeadDim):
    """Calculate theoretical shared memory usage."""
    element_size = 2  # fp16

    # Q block
    smem_q = kBlockM * kHeadDim * element_size

    # K and V blocks
    smem_kv = 2 * kBlockN * kHeadDim * element_size

    # Total
    total_smem = smem_q + smem_kv

    print(f"Config: {kBlockM}×{kBlockN}, HeadDim={kHeadDim}")
    print(f"  Q block: {smem_q / 1024:.1f} KB")
    print(f"  K+V blocks: {smem_kv / 1024:.1f} KB")
    print(f"  Total: {total_smem / 1024:.1f} KB")

    # Check limits
    ampere_limit = 164 * 1024
    hopper_limit = 227 * 1024

    if total_smem <= ampere_limit:
        print(f"  ✓ Fits in Ampere shared memory")
    else:
        print(f"  ✗ Exceeds Ampere limit by {(total_smem - ampere_limit) / 1024:.1f} KB")

    if total_smem <= hopper_limit:
        print(f"  ✓ Fits in Hopper shared memory")
    else:
        print(f"  ✗ Exceeds Hopper limit by {(total_smem - hopper_limit) / 1024:.1f} KB")

    print()

if __name__ == "__main__":
    # Test various configurations
    analyze_memory(64, 64, 128)
    analyze_memory(128, 128, 128)
    analyze_memory(256, 128, 128)
    analyze_memory(256, 256, 128)
```

### Correctness Testing

```python
import torch
from block_sparse_attn import block_sparse_attn_func

def test_correctness(m_block_dim, n_block_dim):
    """Verify custom block size produces correct results."""
    batch, heads, seqlen, headdim = 1, 2, 1024, 128
    device = 'cuda'

    torch.manual_seed(42)

    q = torch.randn(batch * seqlen, heads, headdim, device=device, dtype=torch.float16)
    k = torch.randn(batch * seqlen, heads, headdim, device=device, dtype=torch.float16)
    v = torch.randn(batch * seqlen, heads, headdim, device=device, dtype=torch.float16)

    cu_seqlens = torch.tensor([0, seqlen], device=device, dtype=torch.int32)

    # Dense reference (head_mask_type=0)
    head_mask_type_dense = torch.zeros(heads, dtype=torch.int32, device=device)

    out_dense, _, _ = block_sparse_attn_func(
        q, k, v, cu_seqlens, cu_seqlens,
        head_mask_type=head_mask_type_dense,
        base_blockmask=None,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        m_block_dim=128,
        n_block_dim=128,
        p_dropout=0.0,
        softmax_scale=1.0 / (headdim ** 0.5),
        is_causal=False,
        return_attn_probs=False
    )

    # Block sparse with all blocks active
    num_m_blocks = (seqlen + m_block_dim - 1) // m_block_dim
    num_n_blocks = (seqlen + n_block_dim - 1) // n_block_dim

    base_blockmask = torch.ones(
        (batch, heads, num_m_blocks, num_n_blocks),
        dtype=torch.int32, device=device
    )

    head_mask_type_sparse = torch.ones(heads, dtype=torch.int32, device=device)

    out_sparse, _, _ = block_sparse_attn_func(
        q, k, v, cu_seqlens, cu_seqlens,
        head_mask_type=head_mask_type_sparse,
        base_blockmask=base_blockmask,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        m_block_dim=m_block_dim,
        n_block_dim=n_block_dim,
        p_dropout=0.0,
        softmax_scale=1.0 / (headdim ** 0.5),
        is_causal=False,
        return_attn_probs=False
    )

    # Compare
    diff = (out_dense - out_sparse).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Block size: {m_block_dim}×{n_block_dim}")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")

    if max_diff < 1e-2:
        print(f"  ✓ PASS")
        return True
    else:
        print(f"  ✗ FAIL (difference too large)")
        return False

if __name__ == "__main__":
    # Test various block sizes
    for size in [128, 256, 384, 512]:
        test_correctness(size, size)
        print()
```

---

## Common Pitfalls

### 1. Forgetting to Recompile

**Problem:** Changes to C++ code don't take effect

**Solution:**
```bash
# Force rebuild
pip uninstall block-sparse-attn -y
rm -rf build/ dist/ *.egg-info
pip install -e . --no-build-isolation -v
```

### 2. Block Size Mismatch

**Problem:** `RuntimeError: Invalid block dimensions`

**Solution:** Ensure `m_block_dim` and `n_block_dim` are multiples of `SPARSE_SIZE`:
```python
assert m_block_dim % 128 == 0, "m_block_dim must be multiple of 128"
assert n_block_dim % 128 == 0, "n_block_dim must be multiple of 128"
```

**ViT-Specific:** For 197 tokens (196 patches + CLS), using 128×128 blocks gives 2×2 mask. Using 256×256 would give only 1×1 (no sparsity benefit).

### 3. Shared Memory Overflow

**Problem:** `CUDA error: invalid configuration argument`

**Solution:** Calculate and check shared memory before running:
```python
def check_smem(kBlockM, kBlockN, kHeadDim):
    smem = (kBlockM * kHeadDim + 2 * kBlockN * kHeadDim) * 2  # fp16
    assert smem <= 164 * 1024, f"Exceeds Ampere limit: {smem / 1024:.1f} KB"
```

### 4. Incorrect Mask Dimensions

**Problem:** Attention output is wrong

**Solution:** Double-check mask dimensions:
```python
num_m_blocks = (seqlen_q + m_block_dim - 1) // m_block_dim
num_n_blocks = (seqlen_k + n_block_dim - 1) // n_block_dim

assert base_blockmask.shape == (batch, num_heads, num_m_blocks, num_n_blocks)
```

**ViT Example:**
```python
# ViT-Base: 197 tokens, 128×128 blocks
num_blocks = (197 + 127) // 128  # = 2
base_blockmask.shape  # Should be (batch, 12, 2, 2)
```

---

## Summary

This tutorial covered implementing block-sparse attention for Vision Transformers:

1. **Using existing kernels** with ViT-appropriate block sizes (128×128 for standard resolution)
2. **Adding configurations** for high-resolution ViT (384×384, 512×512 images)
3. **Creating sparsity patterns** suitable for ViT (local+global, strided, learned)
4. **Benchmarking performance** across different patterns and resolutions
5. **Debugging and profiling** techniques for ViT workloads

**Key Takeaways for ViT:**

- ViT uses **bidirectional attention** (no causal masking needed)
- Sequence length is **fixed** based on image resolution and patch size
- Block size should align with sequence length (e.g., 128×128 for 196 patches)
- Common patterns: local windows, global+local (CLS token), learned sparsity
- Typical head dimensions: 64 (ViT-Base/Large) or 128 (ViT-Huge)

**ViT-Specific Optimizations:**

- **Standard ViT (224×224)**: 197 tokens → use 128×128 blocks (2×2 grid)
- **High-res ViT (384×384)**: 577 tokens → use 128×128 or 256×128 blocks
- **Very high-res (512×512)**: 1025 tokens → use 256×256 blocks
- **Pattern selection**: Start with local+global, experiment with learned patterns

**Next Steps:**

- Integrate with your ViT training pipeline
- Experiment with different sparsity patterns per layer
- Consider learnable sparsity masks that adapt during training
- Profile on your target GPU (A100, H100, etc.)
- Test different head configurations (some sparse, some dense)

**Further Reading:**

- [Implementation Guide](./IMPLEMENTATION_GUIDE.md) - Deep dive into implementation details
- [Block Mask Iterator System](./IMPLEMENTATION_GUIDE.md#block-mask-iterator-system) - How masks are processed
- [Kernel Compilation Pipeline](./IMPLEMENTATION_GUIDE.md#kernel-compilation-pipeline) - Build system details
- [Performance Considerations](./IMPLEMENTATION_GUIDE.md#performance-considerations) - Optimization tips
