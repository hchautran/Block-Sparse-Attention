# Block Sparse Attention: Implementation Guide

This guide provides detailed documentation on the implementation of block-sparse attention and instructions for adding custom block sizes to the kernel.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Key Implementation Details](#key-implementation-details)
3. [Block Mask Iterator System](#block-mask-iterator-system)
4. [Kernel Compilation Pipeline](#kernel-compilation-pipeline)
5. [Adding Custom Block Sizes](#adding-custom-block-sizes)
6. [Performance Considerations](#performance-considerations)

## Related Documentation

- [Custom Kernel Tutorial](./CUSTOM_KERNEL_TUTORIAL.md) - Step-by-step tutorial for implementing custom block sizes
- [Main README](./README.md) - Project overview and installation instructions

---

## Architecture Overview

### Layer Structure

The implementation is organized in the following hierarchy:

```
Python Interface (block_sparse_attn/)
    ↓
C++ API Layer (csrc/block_sparse_attn/flash_api.cpp)
    ↓
Launch Template (csrc/block_sparse_attn/src/flash_fwd_launch_template.h)
    ↓
Kernel Implementation (csrc/block_sparse_attn/src/flash_fwd_kernel.h)
    ↓
Block Mask Iterators (csrc/block_sparse_attn/src/flash_blockmask.h)
```

**Key Files:**
- [`block_sparse_attn/__init__.py`](../block_sparse_attn/__init__.py) - Python API entry point
- [`csrc/block_sparse_attn/flash_api.cpp`](../csrc/block_sparse_attn/flash_api.cpp) - C++ API layer ([see Parameter Validation](#parameter-validation))
- [`csrc/block_sparse_attn/src/flash_fwd_launch_template.h`](../csrc/block_sparse_attn/src/flash_fwd_launch_template.h) - Kernel launch logic ([see Launch Templates](#2-launch-templates))
- [`csrc/block_sparse_attn/src/flash_fwd_kernel.h`](../csrc/block_sparse_attn/src/flash_fwd_kernel.h) - Core kernel implementation ([see Kernel Dispatch](#3-kernel-dispatch-logic))
- [`csrc/block_sparse_attn/src/flash_blockmask.h`](../csrc/block_sparse_attn/src/flash_blockmask.h) - Block mask iterators ([see Iterator System](#block-mask-iterator-system))

### Core Components

1. **Kernel Traits** ([`kernel_traits.h`](../csrc/block_sparse_attn/src/kernel_traits.h)): Define block sizes, thread configurations, and memory layouts ([see Template Parameters](#1-template-parameters))
2. **Block Mask Iterators** ([`flash_blockmask.h`](../csrc/block_sparse_attn/src/flash_blockmask.h)): Handle different sparsity patterns ([see Iterator System](#block-mask-iterator-system))
3. **Forward/Backward Kernels** ([`flash_fwd_kernel.h`](../csrc/block_sparse_attn/src/flash_fwd_kernel.h), [`flash_bwd_kernel.h`](../csrc/block_sparse_attn/src/flash_bwd_kernel.h)): Core computation logic
4. **Launch Templates** ([`flash_fwd_launch_template.h`](../csrc/block_sparse_attn/src/flash_fwd_launch_template.h)): Kernel instantiation and dispatch ([see Launch Templates](#2-launch-templates))

---

## Key Implementation Details

### 1. Template Parameters

The system is based on compile-time template parameters for performance:

```cpp
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type>
struct Flash_fwd_kernel_traits {
    static constexpr int kBlockM = kBlockM_;  // Query block size (e.g., 128)
    static constexpr int kBlockN = kBlockN_;  // Key block size (e.g., 128)
    static constexpr int kHeadDim = kHeadDim_; // Head dimension (32, 64, 128)
    static constexpr int kNWarps = kNWarps_;   // Number of warps (4)
    // ...
};
```

**Key Parameters:**
- `kBlockM`: Query (row) block size - must be multiple of 16 for MMA (Matrix Multiply-Accumulate) instructions. This determines how many query tokens are processed together in one thread block. Larger values increase shared memory usage but improve computation efficiency.
- `kBlockN`: Key (column) block size - must be multiple of 16 for MMA instructions. This determines how many key tokens are attended to simultaneously. For ViT, this affects how patch embeddings are grouped during attention computation.
- `kHeadDim`: Attention head dimension (32, 64, or 128). Corresponds to the d_k in the attention formula. ViT typically uses 64 (for ViT-Base with 768 hidden dim / 12 heads) or 128.
- `kNWarps`: Number of warps per thread block (typically 4, can be 8 or 16 for larger blocks). Each warp contains 32 threads. More warps allow more parallelism but increase register pressure.

### 2. Block Sparsity Granularity

The implementation uses a **base granularity of 128×128**:

```cpp
// In csrc/block_sparse_attn/flash_api.cpp
const int SPARSE_SIZE = 128;

// User specifies block dimensions (must be multiples of 128)
int m_block_dim = 128;  // or 256, 384, etc.
int n_block_dim = 128;  // or 256, 384, etc.
```

**File:** [`csrc/block_sparse_attn/flash_api.cpp`](../csrc/block_sparse_attn/flash_api.cpp) (see `mha_varlen_fwd_block()` function)

**Scaling Factors:**
```cpp
// In csrc/block_sparse_attn/src/flash_blockmask.h
this->row_factor = params.m_block_dim / kBlockM;  // How many kBlockM in one m_block_dim
this->col_factor = params.n_block_dim / kBlockN;  // How many kBlockN in one n_block_dim
```

Example: If `m_block_dim=256` and `kBlockM=128`, then `row_factor=2`.

**File:** [`csrc/block_sparse_attn/src/flash_blockmask.h`](../csrc/block_sparse_attn/src/flash_blockmask.h) (see `fwdBlockmask` constructor)

**Detailed Parameter Usage:**

**`m_block_dim` and `n_block_dim`** (user-facing):
- Control the granularity of the sparsity mask
- For ViT with 196 patches (14×14), you might use 128×128 blocks, giving 2×2 = 4 blocks
- Larger blocks mean coarser sparsity control but less overhead
- Must be multiples of 128 (current implementation)

**`head_mask_type`** (per-head configuration):
- Array of integers, one per attention head
- `0`: Dense attention (no sparsity, full N²)
- `>0`: Block-sparse attention with explicit mask
- For ViT: typically use same pattern across heads, or different patterns per head for multi-scale attention

**`base_blockmask`** (sparsity pattern):
- Shape: `(batch, num_heads, num_m_blocks, num_n_blocks)`
- Each element is an integer indicating if that block is active
- For ViT: can encode local windows, global tokens, or learned sparse patterns
- Example: For local+global pattern, set diagonal blocks + first column/row

**`softmax_scale`**:
- Scaling factor applied before softmax, typically `1/√d_k`
- For ViT with d_k=64: scale = 1/8 = 0.125
- Crucial for numerical stability in attention computation

**`cu_seqlens_q` and `cu_seqlens_k`**:
- Cumulative sequence lengths for variable-length sequences in a batch
- For ViT with fixed image size: all sequences have same length (e.g., [0, 197, 394, ...] for batch with CLS token)
- Allows efficient batching without padding overhead

### 3. Kernel Dispatch Logic

The forward pass dispatches based on `head_mask_type`:

```cpp
// In csrc/block_sparse_attn/src/flash_fwd_kernel.h: compute_block_attn()
const int head_mask_type = params.head_mask_type[bidh];

if (head_mask_type == 0) {
    // Dense attention - full N² complexity
    // Use this as a baseline or for heads that need full attention
    compute_attn_1rowblock<Kernel_traits, ...>(params, bidb, bidh, m_block);

} else if (head_mask_type > 0) {
    // Block sparse attention - uses explicit mask from base_blockmask
    // For ViT: This is the main mode for sparse attention patterns
    // The mask defines which blocks of patches attend to each other
    compute_block_attn_1rowblock<Kernel_traits, ..., false, false>(
        params, bidb, bidh, m_block, n_block_min, n_block_max
    );
}
```

**File:** [`csrc/block_sparse_attn/src/flash_fwd_kernel.h`](../csrc/block_sparse_attn/src/flash_fwd_kernel.h) (see `flash_fwd_block_kernel()` and `compute_block_attn()`)

**For ViT Implementation:**
- Use `head_mask_type > 0` for all heads with custom sparsity patterns
- Use `head_mask_type = 0` for specific heads if you want full attention (e.g., for global reasoning)
- See [Custom Kernel Tutorial](./CUSTOM_KERNEL_TUTORIAL.md#quick-start-using-existing-kernels-for-vit) for practical examples

---

## Block Mask Iterator System

### Iterator Architecture

The iterator handles block-sparse patterns defined by your mask:

```cpp
template<bool Is_streaming, bool Is_exact_streaming>
class fwdIterator {};

// For ViT, we use the block-sparse iterator:
template<> struct fwdIterator<false, false>: public fwdBlockmask {};
```

### fwdBlockmask Iterator

Handles explicit block-sparse patterns (primary mode for ViT).

**File:** [`csrc/block_sparse_attn/src/flash_blockmask.h`](../csrc/block_sparse_attn/src/flash_blockmask.h)

```cpp
class fwdBlockmask {
    __device__ fwdBlockmask(const Params &params, const BlockInfo &binfo,
                             int kBlockM, int kBlockN, int batch_idx,
                             int head_idx, int loop_step_idx,
                             int n_block_min, int n_block_max) {
        // Calculate scaling factors between user-facing and kernel block sizes
        this->row_factor = params.m_block_dim / kBlockM;
        this->col_factor = params.n_block_dim / kBlockN;

        // Position pointer to current query block's mask row
        // For ViT: batch_idx selects the image, head_idx selects the attention head
        // loop_step_idx indicates which query block we're processing
        blockmask_ptr = params.blockmask +
            (batch_idx * params.num_blocksparse_heads + mask_type - 1)
            * (params.seqlen_q_rounded / m_block_dim)  // Number of query blocks
            * (params.seqlen_k_rounded / n_block_dim)  // Number of key blocks
            + (loop_step_idx / row_factor)  // Current query block row
            * (params.seqlen_k_rounded / n_block_dim);  // Stride to next row
    }

    // Query if a block column is active
    __device__ int mask_val(int block_col_idx) const {
        if (block_col_idx > max_block_idx || block_col_idx < 0) return -1;

        // Convert from kBlockN granularity to m_block_dim granularity
        int real_block_idx = block_col_idx / col_factor;
        int block_col_offset = block_col_idx % col_factor;

        // Read mask value
        int mask_val = blockmask_ptr[real_block_idx];

        // Transform back to kBlockN granularity
        return mask_val == -1 ? -1 :
               col_factor * mask_val + col_factor - 1 - block_col_offset;
    }
};
```

**Key Insight:** The mask is stored at `m_block_dim × n_block_dim` granularity (user-specified, coarse), but the kernel iterates at `kBlockM × kBlockN` granularity (hardware-optimized, fine). The `col_factor` and `row_factor` bridge these two levels.

**For ViT Example:**
- Image: 224×224, patch size 16×16 → 196 patches
- Block size: 128×128
- num_m_blocks = num_n_blocks = ⌈196/128⌉ = 2
- Your mask is 2×2, controlling attention between 4 groups of ~98 patches each


### Backward Pass Iterators

The backward pass uses **transposed masks** because it iterates column-wise (over key blocks) instead of row-wise.

**File:** [`csrc/block_sparse_attn/src/flash_blockmask.h`](../csrc/block_sparse_attn/src/flash_blockmask.h)

```cpp
class bwdBlockmask {
    __device__ bwdBlockmask(...) {
        // Position to column's mask (transposed)
        // During backprop, we compute gradients w.r.t. keys by iterating over key blocks
        blockmask_ptr = params.blockmask +
            (batch_idx * params.num_blocksparse_heads + mask_type - 1)
            * (params.seqlen_k_rounded / n_block_dim)  // Now iterating over K dimension
            * (params.seqlen_q_rounded / m_block_dim)  // Q dimension becomes inner
            + (loop_step_idx / col_factor)  // Current key block column
            * (params.seqlen_q_rounded / m_block_dim);  // Stride
    }

    // Query which query blocks attend to this key block
    __device__ int mask_val(int block_row_idx) const;
};
```

**Python mask conversion:**
```python
# Forward pass: row-major mask (query blocks iterate over key blocks)
row_blockmask = convert_blockmask_row_reverse(base_blockmask)

# Backward pass: transposed column-major mask (key blocks iterate over query blocks)
col_blockmask = convert_blockmask_col_reverse(base_blockmask)

# For ViT: Both conversions are automatically handled by block_sparse_attn_func
# when you pass base_blockmask
```

---

## Kernel Compilation Pipeline

### 1. Kernel Generation Script

The [`generate_kernels.py`](../csrc/block_sparse_attn/src/generate_kernels.py) script generates kernel instantiations:

```python
# In csrc/block_sparse_attn/src/generate_kernels.py
SM = [80]  # Compute capability (80 for A100, 86 for A6000, 89 for RTX 4090, 90 for H100)
HEAD_DIMENSIONS = [32, 64, 128]  # Attention head dimensions supported
DTYPE_MAP = {"fp16": "cutlass::half_t", "bf16": "cutlass::bfloat16_t"}

# For ViT, you typically need:
# - HEAD_DIMENSIONS: 64 (ViT-Base/Large) or 128 (ViT-Huge)
# - DTYPE: fp16 for inference, bf16 for training on Ampere+

# Generates files like:
# flash_fwd_block_hdim64_fp16_sm80.cu
# flash_fwd_block_hdim64_bf16_sm80.cu
# flash_fwd_block_hdim128_fp16_sm80.cu
# ...
```

Each generated file contains:

```cpp
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template<>
void run_mha_fwd_block_<cutlass::half_t, 32, true>(
    Flash_fwd_params &params, cudaStream_t stream
) {
    run_mha_fwd_block_hdim32<cutlass::half_t, true>(params, stream);
}

}
```

### 2. Launch Templates

The launch template selects block configurations based on head dimension.

**File:** [`csrc/block_sparse_attn/src/flash_fwd_launch_template.h`](../csrc/block_sparse_attn/src/flash_fwd_launch_template.h)

```cpp
// In csrc/block_sparse_attn/src/flash_fwd_launch_template.h
template<typename T, bool Is_causal>
void run_mha_fwd_block_hdim32(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 32;
    BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_fwd_block<
            Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>,
            Is_dropout, Is_causal
        >(params, stream);
    });
}

template<typename T>
void run_mha_fwd_block_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x = cc_major == 8 && cc_minor > 0;

    // For ViT: No causal masking needed, use Is_dropout for training vs inference
    BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if constexpr(!Is_dropout) {
            // Inference or no dropout training
            if (is_sm8x) {
                // 128x32 optimized for Ampere+ (A100, A6000, RTX 4090)
                // Wide blocks improve memory throughput for bidirectional attention
                run_flash_fwd_block<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, ...>, ...>;
            } else {
                // Fallback for older architectures
                run_flash_fwd_block<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, ...>, ...>;
            }
        } else {
            // Training with dropout
            run_flash_fwd_block<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, ...>, ...>;
        }
    });
}
```

### 3. Kernel Invocation

```cpp
template<typename Kernel_traits, bool Is_dropout>
void run_flash_fwd_block(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;

    // Grid dimensions:
    // - X: number of query blocks (for ViT with 196 patches and kBlockM=128: 2 blocks)
    // - Y: batch size (number of images)
    // - Z: number of attention heads
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1)
                           / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);

    // Each block uses Kernel_traits::kNThreads threads (typically 128 = 4 warps × 32 threads)
    auto kernel = &flash_fwd_block_kernel<Kernel_traits, Is_dropout, ...>;

    // Set shared memory size if needed (required when > 48KB)
    if (smem_size >= 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size);
    }

    // Launch kernel
    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
}
```

---

## Adding Custom Block Sizes

### Step 1: Understand Constraints

**Hard Constraints:**
1. `kBlockM` and `kBlockN` must be multiples of 16 (MMA instruction requirement - hardware limitation)
2. User-facing `m_block_dim` and `n_block_dim` must be multiples of 128 (current implementation, can be changed to 64)
3. `m_block_dim` must be a multiple of `kBlockM` (ensures row_factor is an integer)
4. `n_block_dim` must be a multiple of `kBlockN` (ensures col_factor is an integer)

**Soft Constraints (performance):**
1. Shared memory usage must be < 164KB (Ampere), < 227KB (Hopper) - exceeding causes launch failure
2. Register usage should allow at least 1 CTA per SM - too many registers reduces occupancy
3. Block sizes should align with warp size (32) for efficiency - reduces thread divergence

**For ViT Considerations:**
- Sequence length is typically fixed (196, 256, or 577 patches)
- Choose block sizes that divide sequence length evenly to avoid waste
- For 196 patches: 128×128 gives clean 2×2 block structure

### Step 2: Calculate Shared Memory Usage

The shared memory requirement is:

```cpp
// For forward pass
static constexpr int kSmemQSize = kBlockM * kHeadDim * sizeof(Element);
static constexpr int kSmemKVSize = 2 * kBlockN * kHeadDim * sizeof(Element);
static constexpr int kSmemSize = kSmemQSize + kSmemKVSize;  // If not sharing Q/K memory

// Example: kBlockM=128, kBlockN=128, kHeadDim=128, fp16
// kSmemQSize = 128 * 128 * 2 = 32 KB
// kSmemKVSize = 2 * 128 * 128 * 2 = 64 KB
// kSmemSize = 96 KB
```

### Step 3: Add New Kernel Trait Instantiation

Create a new configuration in [`flash_fwd_launch_template.h`](../csrc/block_sparse_attn/src/flash_fwd_launch_template.h):

**See also:** [Custom Kernel Tutorial - Adding Custom Block Sizes](./CUSTOM_KERNEL_TUTORIAL.md#tutorial-1-adding-192192-block-support)

```cpp
template<typename T, bool Is_causal>
void run_mha_fwd_block_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;

    BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if constexpr(!Is_dropout) {
            // ADD YOUR CUSTOM CONFIGURATION HERE
            // Example: 256x128 blocks for large sequence lengths
            if (params.seqlen_q > 4096 && params.seqlen_k > 4096) {
                run_flash_fwd_block<
                    Flash_fwd_kernel_traits<Headdim, 256, 128, 8, false, false, T>,
                    //                               ^^^  ^^^  ^
                    //                              kBlockM  kBlockN  kNWarps
                    Is_dropout, Is_causal
                >(params, stream);
            }
            // ... existing configurations ...
        }
    });
}
```

### Step 4: Validate Constraints

Add validation in [`flash_api.cpp`](../csrc/block_sparse_attn/flash_api.cpp):

```cpp
// In mha_varlen_fwd_block()
if (has_blockmask) {
    TORCH_CHECK(m_block_dim % SPARSE_SIZE == 0,
                "m_block_dim must be a multiple of 128");
    TORCH_CHECK(n_block_dim % SPARSE_SIZE == 0,
                "n_block_dim must be a multiple of 128");

    // ADD: Validate against kernel block sizes
    TORCH_CHECK(m_block_dim >= 128 && m_block_dim <= 512,
                "m_block_dim must be between 128 and 512");
    TORCH_CHECK(n_block_dim >= 128 && n_block_dim <= 512,
                "n_block_dim must be between 128 and 512");
}
```

### Step 5: Recompile

```bash
# Clean previous build
rm -rf build/ dist/ *.egg-info

# Regenerate kernels if you modified generate_kernels.py
cd csrc/block_sparse_attn/src
python generate_kernels.py

# Build
cd ../../../
pip install -e . --no-build-isolation
```

### Step 6: Test Custom Block Size

```python
import torch
from block_sparse_attn import block_sparse_attn_func

# Create test inputs
batch, heads, seqlen, headdim = 2, 8, 2048, 128
q = torch.randn(batch * seqlen, heads, headdim, device='cuda', dtype=torch.float16)
k = torch.randn(batch * seqlen, heads, headdim, device='cuda', dtype=torch.float16)
v = torch.randn(batch * seqlen, heads, headdim, device='cuda', dtype=torch.float16)

cu_seqlens = torch.tensor([0, seqlen, 2*seqlen], device='cuda', dtype=torch.int32)

# Use custom block dimensions (must be multiples of 128)
m_block_dim = 256  # Custom query block size
n_block_dim = 256  # Custom key block size

# Create block mask for custom dimensions
num_m_blocks = (seqlen + m_block_dim - 1) // m_block_dim
num_n_blocks = (seqlen + n_block_dim - 1) // n_block_dim

base_blockmask = torch.ones(
    (batch, 1, num_m_blocks, num_n_blocks),
    dtype=torch.int32, device='cuda'
)

head_mask_type = torch.tensor([1] * heads, dtype=torch.int32, device='cuda')

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

print(f"Success with custom block size {m_block_dim}x{n_block_dim}!")
```

---

## Example: Adding 256×256 Block Support

Here's a complete example of adding support for 256×256 user-facing blocks.

**See also:** [Custom Kernel Tutorial](./CUSTOM_KERNEL_TUTORIAL.md) for step-by-step instructions

### 1. Modify `flash_fwd_launch_template.h`

**File:** [`csrc/block_sparse_attn/src/flash_fwd_launch_template.h`](../csrc/block_sparse_attn/src/flash_fwd_launch_template.h)

```cpp
template<typename T, bool Is_causal>
void run_mha_fwd_block_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x = cc_major == 8 && cc_minor > 0;

    BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if constexpr(!Is_dropout) {
            // NEW: Support for 256×256 user blocks with 128×128 kernel blocks
            // This means row_factor=2, col_factor=2
            // The iterator will handle the mapping automatically

            if (is_sm8x) {
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

**Note:** You don't need to modify the kernel itself! The existing kernel with `kBlockM=128, kBlockN=128` can handle user-facing `m_block_dim=256, n_block_dim=256` through the `row_factor=2, col_factor=2` mechanism in the iterator.

### 2. Test with 256×256 Blocks

```python
# The kernel internally uses 128×128, but you specify 256×256 at Python level
m_block_dim = 256
n_block_dim = 256

# The mask should be at 256×256 granularity
num_m_blocks = (seqlen + 255) // 256  # Ceiling division
num_n_blocks = (seqlen + 255) // 256

base_blockmask = create_custom_mask(batch, num_heads, num_m_blocks, num_n_blocks)

# The iterator will automatically map:
# - Each 256×256 user block → four 128×128 kernel blocks
# - row_factor = 2, col_factor = 2
```

---

## Performance Considerations

### 1. Block Size Selection

**Small blocks (64×64, 128×64):**
- ✅ Lower shared memory usage
- ✅ More CTAs per SM (better occupancy)
- ❌ More kernel launches needed for large sequences
- ❌ Higher overhead from block boundaries

**Medium blocks (128×128):**
- ✅ Balanced shared memory vs computation
- ✅ Good for most workloads
- Current default configuration

**Large blocks (256×128, 256×256):**
- ✅ Fewer kernel launches for long sequences
- ✅ Better for high-end GPUs (A100, H100)
- ❌ Higher shared memory usage
- ❌ May reduce occupancy

### 2. Architecture-Specific Tuning

**Ampere (SM80, SM86 - A100, A6000, RTX 4090):**
- Shared memory: up to 164 KB per block
- For ViT: Use wide blocks (128×32, 128×64) for better memory throughput
- Good balance between occupancy and compute efficiency

**Hopper (SM90 - H100):**
- Shared memory: up to 227 KB per block
- Can support larger blocks (256×128, 256×256)
- Better tensor core utilization with larger blocks
- For ViT with high resolution (e.g., 1024+ patches): Use 256×256 blocks

### 3. Sparsity Pattern Impact

**Dense regions:**
- Use larger blocks to amortize overhead
- For ViT: Local attention windows benefit from larger blocks

**Highly sparse regions:**
- Smaller blocks reduce wasted computation
- For ViT: If only attending to CLS token + local patches, smaller blocks are better
- Consider dynamic block size selection based on sparsity ratio

**Common ViT Sparsity Patterns:**
- **Local windows**: Patches only attend to nearby patches (band diagonal mask)
- **Strided attention**: Skip certain spatial distances (chess board pattern)
- **Global + Local**: All patches attend to CLS token + local neighborhood
- **Learned sparsity**: Use Top-K or learned masks per layer

### 4. Measuring Performance

```python
import torch.utils.benchmark as benchmark

def measure_kernel(m_block, n_block, seqlen):
    # Setup tensors...

    timer = benchmark.Timer(
        stmt='block_sparse_attn_func(q, k, v, cu_seqlens, cu_seqlens, ...)',
        globals={'q': q, 'k': k, 'v': v, ...}
    )

    result = timer.blocked_autorange(min_run_time=1.0)
    print(f"Block {m_block}×{n_block}: {result.median*1000:.3f} ms")

# Compare different block sizes
# For ViT with 196 patches, test smaller configurations
for size in [128, 256]:
    measure_kernel(size, size, seqlen=196)
```

---

## Advanced Topics

### 1. Supporting Non-Standard Block Sizes

To support block sizes that aren't multiples of 128:

**File:** [`csrc/block_sparse_attn/flash_api.cpp`](../csrc/block_sparse_attn/flash_api.cpp)

```cpp
// Modify SPARSE_SIZE in csrc/block_sparse_attn/flash_api.cpp
const int SPARSE_SIZE = 64;  // Reduce to 64

// Update validation in mha_varlen_fwd_block()
TORCH_CHECK(m_block_dim % SPARSE_SIZE == 0,
            "m_block_dim must be a multiple of 64");
```

**Caveat:** This requires ensuring all kernels use compatible block sizes.

**See:** [Custom Kernel Tutorial - Adding 192×192 Support](./CUSTOM_KERNEL_TUTORIAL.md#tutorial-1-adding-192192-block-support)

### 2. Dynamic Block Size Selection

**File:** [`csrc/block_sparse_attn/src/flash_fwd_launch_template.h`](../csrc/block_sparse_attn/src/flash_fwd_launch_template.h)

```cpp
// In launch template - ViT-specific selection
if (params.seqlen_q <= 256) {
    // ViT-Small/Base (14×14 = 196 patches + 1 CLS)
    // Use smaller blocks for efficient small sequence handling
    run_flash_fwd_block<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, ...>, ...>;
} else if (params.seqlen_q <= 1024) {
    // ViT-Large (16×16 = 256 patches or higher resolution)
    run_flash_fwd_block<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, ...>, ...>;
} else {
    // High-resolution ViT (32×32 = 1024 patches or more)
    run_flash_fwd_block<Flash_fwd_kernel_traits<Headdim, 256, 128, 8, ...>, ...>;
}
```

### 3. Mixed Block Sizes

Currently, all heads use the same block size. To support per-head block sizes (useful for multi-scale ViT):

1. Extend `Flash_fwd_params` with per-head block dimensions
2. Modify iterators to read per-head block configurations
3. Launch kernels with different grid dimensions per head

**ViT Use Case:**
- First few heads: small blocks for fine-grained local patterns
- Middle heads: medium blocks for mid-range interactions
- Last heads: large blocks or dense for global reasoning

---

## Troubleshooting

### Problem: "CUDA error: invalid configuration argument"

**Cause:** Shared memory exceeds limit

**Solution:**
```python
# Check shared memory usage
smem = (kBlockM * kHeadDim + 2 * kBlockN * kHeadDim) * sizeof_element
print(f"Shared memory required: {smem / 1024:.1f} KB")

# Must be < 164 KB for Ampere, < 227 KB for Hopper
```

### Problem: Performance degradation with custom blocks

**Diagnosis:**
1. Check occupancy: `nvprof --metrics achieved_occupancy`
2. Check shared memory usage: `nvprof --metrics shared_memory_usage`
3. Profile with `nsys` to identify bottlenecks

**Common issues:**
- Register spills (reduce `kBlockM`, `kBlockN`, or `kHeadDim`)
- Low occupancy (reduce shared memory usage)
- Bank conflicts (check memory layout patterns)

### Problem: Incorrect outputs with custom block sizes

**Debug checklist:**
1. Verify `row_factor` and `col_factor` calculations
2. Check mask pointer arithmetic in iterators
3. Validate mask dimensions match block size
4. Test with dense attention first (head_mask_type=0)

---

## Summary

Adding custom block sizes involves:

1. **Understanding** the two-level granularity: user-facing (`m_block_dim`, `n_block_dim`) vs kernel (`kBlockM`, `kBlockN`)
2. **Calculating** scaling factors: `row_factor = m_block_dim / kBlockM`, `col_factor = n_block_dim / kBlockN`
3. **Selecting** kernel configurations in launch templates based on workload characteristics
4. **Validating** shared memory constraints and performance characteristics
5. **Testing** thoroughly with different sparsity patterns and sequence lengths

The iterator system automatically handles the mapping between user-specified and kernel block sizes, making it straightforward to support larger user-facing blocks without kernel modifications.

**For ViT Implementation:**
- **Standard ViT (196-256 patches)**: Use 128×128 blocks for 2×2 mask structure
- **High-resolution ViT (1024+ patches)**: Use 256×256 blocks for coarser control
- **Efficiency-focused**: Use 64×64 or 128×64 for highly sparse patterns
- **Memory-constrained**: Smaller blocks reduce peak memory usage

**Example ViT Configuration:**
```python
# ViT-Base: 224×224 image, 16×16 patches → 196 patches + 1 CLS = 197 tokens
# Use 128×128 blocks → 2×2 grid
# Mask shape: (batch, 12 heads, 2, 2)
```

