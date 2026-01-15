# Flash Forward Kernel Implementation Explained

This document provides a detailed explanation of how the Flash Attention forward pass is implemented in [`csrc/block_sparse_attn/src/flash_fwd_kernel.h`](../csrc/block_sparse_attn/src/flash_fwd_kernel.h).

## Table of Contents

1. [Overview](#overview)
2. [Kernel Entry Point](#kernel-entry-point)
3. [Main Algorithm Flow](#main-algorithm-flow)
4. [Key Functions Explained](#key-functions-explained)
5. [Memory Layout and Tensors](#memory-layout-and-tensors)
6. [The Flash Attention Algorithm](#the-flash-attention-algorithm)
7. [Block-Sparse Optimization](#block-sparse-optimization)
8. [Complete Execution Flow](#complete-execution-flow)

## Related Documentation

- [C++ Techniques Guide](./CPP_TECHNIQUES_GUIDE.md) - Understanding C++ and CUDA concepts
- [Implementation Guide](./IMPLEMENTATION_GUIDE.md) - Architecture overview
- [Custom Kernel Tutorial](./CUSTOM_KERNEL_TUTORIAL.md) - Practical examples

**Prerequisites:** Familiarity with CUDA programming and C++ templates. If these are new to you, start with the [C++ Techniques Guide](./CPP_TECHNIQUES_GUIDE.md).

---

## Overview

### What is Flash Attention?

Flash Attention is an algorithm that computes exact attention (same result as standard attention) while:
- Using **O(N)** memory instead of **O(N²)**
- Being **faster** than standard attention
- Enabling much longer sequence lengths

**Key Idea:** Instead of materializing the full N×N attention matrix, process it in blocks and keep running statistics.

### File Structure

The implementation is split across several files:

```
flash_fwd_kernel.h (main kernel logic)
  ├─ softmax_rescale_o()          - Online softmax with rescaling
  ├─ compute_attn_1rowblock()     - Process one query block (dense)
  ├─ compute_block_attn_1rowblock() - Process one query block (sparse)
  └─ compute_block_attn()         - Kernel entry point & dispatch

flash_fwd_launch_template.h
  └─ flash_fwd_block_kernel()     - __global__ kernel wrapper
```

---

## Kernel Entry Point

### The Global Kernel

**File:** [`csrc/block_sparse_attn/src/flash_fwd_launch_template.h`](../csrc/block_sparse_attn/src/flash_fwd_launch_template.h)

```cpp
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, ...>
__global__ void flash_fwd_block_kernel(Flash_fwd_params params) {
    FLASH_NAMESPACE::compute_block_attn<Kernel_traits, Is_dropout, Is_causal, ...>(params);
}
```

**What this does:**
- `__global__` = Entry point from CPU to GPU
- Each CUDA block processes **one query block** for **one head** in **one batch item**
- Immediately calls `compute_block_attn()`

### Kernel Launch Configuration

```cpp
// From flash_fwd_launch_template.h
const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
dim3 grid(num_m_block, params.b, params.h);
//        ^^^^^^^^^^^  ^^^^^^^^  ^^^^^^^^
//        Query blocks  Batches   Heads

kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
//            ^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^
//            128 threads            Shared memory size
```

**Grid dimensions:**
- **X**: Number of query blocks (e.g., 2 for 197 tokens with 128-size blocks)
- **Y**: Batch size
- **Z**: Number of attention heads

**Example for ViT-Base:**
- 197 tokens, 128 block size → 2 query blocks
- Batch size 4
- 12 heads
- **Total: 2 × 4 × 12 = 96 CUDA blocks** launched

---

## Main Algorithm Flow

### High-Level Overview

```
For each query block Q[m]:
  1. Load Q[m] into shared memory
  2. Initialize accumulator acc_o = 0, max = -inf, sum = 0
  3. For each key block K[n] (in reverse order):
       a. Load K[n], V[n] into shared memory
       b. Compute scores S = Q @ K^T
       c. Apply masking (causal, local, block-sparse)
       d. Online softmax: update max, rescale acc_o, update sum
       e. Apply dropout (if enabled)
       f. Accumulate: acc_o += softmax(S) @ V
  4. Final scaling: acc_o /= sum
  5. Write output O[m] and log-sum-exp (LSE)
```

### The Dispatch Function

**File:** [`csrc/block_sparse_attn/src/flash_fwd_kernel.h:1270`](../csrc/block_sparse_attn/src/flash_fwd_kernel.h)

```cpp
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, ...>
inline __device__ void compute_block_attn(const Params &params) {
    // Get block indices
    const int m_block = blockIdx.x;  // Which query block
    const int bidb = blockIdx.y;     // Which batch
    const int bidh = blockIdx.z;     // Which head

    // Dispatch based on head_mask_type
    const int head_mask_type = params.head_mask_type[bidh];

    if (head_mask_type == 0) {
        // Dense attention - no sparsity
        compute_attn_1rowblock<Kernel_traits, ...>(params, bidb, bidh, m_block);
    }
    else if (head_mask_type > 0) {
        // Block-sparse attention with explicit mask
        compute_block_attn_1rowblock<Kernel_traits, ..., false, false>(
            params, bidb, bidh, m_block
        );
    }
    else {
        // Streaming attention (not used for ViT)
        compute_block_attn_1rowblock<Kernel_traits, ..., true, Is_exact_streaming>(
            params, bidb, bidh, m_block
        );
    }
}
```

**Key Points:**
- Each CUDA block processes **one query block** (m_block)
- `head_mask_type` determines the attention pattern:
  - `0` = Dense (full N²)
  - `>0` = Block-sparse (use mask)
  - `<0` = Streaming (special pattern)
- For ViT, use `head_mask_type > 0`

---

## Key Functions Explained

### 1. Online Softmax with Rescaling

**File:** [`csrc/block_sparse_attn/src/flash_fwd_kernel.h:34`](../csrc/block_sparse_attn/src/flash_fwd_kernel.h)

```cpp
template<bool Is_first, bool Check_inf, typename Tensor0, typename Tensor1, typename Tensor2>
inline __device__ void softmax_rescale_o(
    Tensor0 &scores,       // Current attention scores S = Q @ K^T
    Tensor1 &scores_max,   // Running maximum
    Tensor1 &scores_sum,   // Running sum of exp
    Tensor2 &acc_o,        // Accumulated output
    float softmax_scale_log2  // 1/sqrt(d_k) in log2 scale
) {
    if (Is_first) {
        // First key block: initialize
        reduce_max</*zero_init=*/true>(scores, scores_max);
        scale_apply_exp2(scores, scores_max, softmax_scale_log2);
        reduce_sum(scores, scores_sum);
    } else {
        // Subsequent blocks: online update
        Tensor scores_max_prev = make_fragment_like(scores_max);
        copy(scores_max, scores_max_prev);

        // Update maximum
        reduce_max</*zero_init=*/false>(scores, scores_max);

        // Rescale previous accumulator
        for (int mi = 0; mi < size(scores_max); ++mi) {
            float scores_max_cur = scores_max(mi);
            float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);

            scores_sum(mi) *= scores_scale;

            // Rescale all columns of acc_o
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
                acc_o_rowcol(mi, ni) *= scores_scale;
            }
        }

        // Compute exp of current scores
        scale_apply_exp2(scores, scores_max, softmax_scale_log2);

        // Update sum
        Tensor scores_sum_cur = make_fragment_like(scores_sum);
        reduce_sum(scores, scores_sum_cur);
        for (int mi = 0; mi < size(scores_sum); ++mi) {
            scores_sum(mi) += scores_sum_cur(mi);
        }
    }
}
```

**The Math Behind This:**

Standard softmax:
```
softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
```

Online softmax (Flash Attention trick):
```
max_new = max(max_old, max_current)
scale = exp(max_old - max_new)
sum_new = sum_old * scale + sum_current
output_new = output_old * scale + output_current
```

**Why This Works:**
- We can update the softmax incrementally without storing all scores
- Previous outputs are rescaled when we see a new maximum
- This is mathematically equivalent to computing full softmax

**Example:**
```
Block 1: scores = [1, 2, 3]
  max = 3, sum = exp(-2) + exp(-1) + exp(0) = 3.27
  output = [0.09, 0.24, 0.67]

Block 2: scores = [4, 5, 6]
  max_new = max(3, 6) = 6
  scale = exp(3 - 6) = 0.05  <- Rescale previous output
  output_old *= 0.05 = [0.004, 0.012, 0.033]
  output_new = [0.09, 0.24, 0.67]  <- From block 2
  output_final = [0.004, 0.012, 0.033, 0.09, 0.24, 0.67]
```

### 2. Main Attention Computation (Dense Version)

**File:** [`csrc/block_sparse_attn/src/flash_fwd_kernel.h:150`](../csrc/block_sparse_attn/src/flash_fwd_kernel.h)

Let me break down the dense attention function:

```cpp
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, ...>
inline __device__ void compute_attn_1rowblock(
    const Params &params, const int bidb, const int bidh, const int m_block
) {
    // 1. Setup
    extern __shared__ char smem_[];
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;  // e.g., 128
    constexpr int kBlockN = Kernel_traits::kBlockN;  // e.g., 128
    constexpr int kHeadDim = Kernel_traits::kHeadDim;  // e.g., 64
```

**Step 1: Memory Setup**

```cpp
    // Create tensor views for global memory
    Tensor gQ = make_tensor(make_gmem_ptr(q_ptr + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(q_row_stride, _1{}));

    // Create tensor views for shared memory
    Tensor sQ = make_tensor(make_smem_ptr(smem_),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + size(sQ),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK),
                            typename Kernel_traits::SmemLayoutKV{});
```

**Visual Layout:**
```
Shared Memory:
┌───────────────────────┐
│  sQ (kBlockM×kHeadDim)│  Query block
├───────────────────────┤
│  sK (kBlockN×kHeadDim)│  Key block
├───────────────────────┤
│  sV (kBlockN×kHeadDim)│  Value block
└───────────────────────┘
```

**Step 2: Load Query Block**

```cpp
    // Load Q into shared memory (done once)
    copy(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, actual_seqlen_q - m_block * kBlockM);

    // Initialize accumulator
    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    clear(acc_o);  // Set to zero

    Tensor scores_max = make_tensor<ElementAccum>(Shape<Int<2 * size<1>(acc_o)>>{});
    Tensor scores_sum = make_fragment_like(scores_max);
    clear(scores_max);  // Will be set to -inf
    clear(scores_sum);  // Set to zero
```

**Step 3: Loop Over Key Blocks**

```cpp
    // Iterate backwards through key blocks
    for (int n_block = n_block_max - 1; n_block >= n_block_min; --n_block) {

        // 3a. Load K and V blocks
        copy(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);
        copy(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
        cp_async_fence();  // Fence async copies

        // 3b. Compute S = Q @ K^T
        Tensor acc_s = partition_fragment_C(tiled_mma,
                                           Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);
        gemm(tiled_mma, tSrQ, tSrK, acc_s);  // Matrix multiply
```

**What `gemm()` does:**
- Performs `acc_s = tSrQ @ tSrK^T`
- Uses GPU tensor cores for acceleration
- Result is attention scores for this Q×K pair

```cpp
        // 3c. Apply masking
        if (Is_causal) {
            apply_mask_local(scores, n_block * kBlockN, actual_seqlen_k,
                           m_block * kBlockM, actual_seqlen_q, ...);
        }
```

**Masking Example (Causal):**
```
Query block m=0 (tokens 0-127):
  Can attend to Key blocks: [0-127, 128-255, ...]

Query block m=1 (tokens 128-255):
  Can attend to Key blocks: [0-127, 128-255, 256-383, ...]
  But tokens 128-255 cannot attend to future tokens within blocks
```

```cpp
        // 3d. Online softmax and rescale
        if (n_block == n_block_max - 1) {
            // First block
            softmax_rescale_o</*Is_first=*/true>(
                scores, scores_max, scores_sum, acc_o, scale_softmax_log2
            );
        } else {
            // Subsequent blocks
            softmax_rescale_o</*Is_first=*/false>(
                scores, scores_max, scores_sum, acc_o, scale_softmax_log2
            );
        }

        // 3e. Convert scores to fp16/bf16
        Tensor rP = convert_type<Element>(scores);

        // 3f. Apply dropout (if enabled)
        if (Is_dropout) {
            apply_dropout(tOrP, p_dropout_in_uint8_t, seed, offset, ...);
        }

        // 3g. Accumulate: acc_o += softmax(S) @ V
        cp_async_wait<0>();  // Wait for V to be loaded
        __syncthreads();

        gemm_A_in_regs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, ...);
        //             ^^^^^  ^^^^  ^^^^^
        //             out    P     V
    }
```

**After the loop:**
- `acc_o` contains the weighted sum of all values
- `scores_sum` contains the normalization factor
- `scores_max` contains the maximum score seen

**Step 4: Final Normalization and Write Output**

```cpp
    // Normalize by sum
    for (int mi = 0; mi < size(lse); ++mi) {
        float sum = scores_sum(mi);
        float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1.f / sum;
        lse(mi) = (sum == 0.f || sum != sum) ?
                  INFINITY : scores_max(mi) * softmax_scale + __logf(sum);

        // Scale output
        for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
            acc_o_rowcol(mi, ni) *= inv_sum;
        }
    }

    // Convert to fp16/bf16
    Tensor rO = convert_type<Element>(acc_o);

    // Write to shared memory
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});
    copy(smem_tiled_copy_O, taccOrO, taccOsO);

    __syncthreads();

    // Write to global memory
    copy(gmem_tiled_copy_O, tOsO, tOgO, tOcO, tOpO, actual_seqlen_q - m_block * kBlockM);

    // Write log-sum-exp (needed for backward pass)
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = get<0>(taccOcO_row(mi));
        if (row < actual_seqlen_q - m_block * kBlockM) {
            gLSE(row) = lse(mi);
        }
    }
}
```

---

## Memory Layout and Tensors

### CuTe Tensor Abstraction

The code uses NVIDIA's **CuTe** library for tensor operations. CuTe provides:
- Type-safe tensor views
- Compile-time layout optimization
- Unified interface for different memory types

### Example: Creating a Tensor

```cpp
// Global memory tensor (Q matrix)
Tensor gQ = make_tensor(
    make_gmem_ptr(q_ptr + offset),  // Pointer to data
    Shape<Int<kBlockM>, Int<kHeadDim>>{},  // Shape: 128×64
    make_stride(q_row_stride, _1{})  // Strides: row-major
);
```

**What this means:**
- Points to a 128×64 region of global memory
- Row-major layout: consecutive elements in a row are adjacent in memory
- `q_row_stride` allows for padded rows

### Shared Memory Layout

```cpp
// Shared memory for Q
Tensor sQ = make_tensor(
    make_smem_ptr(smem_),  // Shared memory pointer
    typename Kernel_traits::SmemLayoutQ{}  // Layout with swizzling
);
```

**Swizzling:** Memory layout optimization to avoid bank conflicts.

```
Without swizzling:        With swizzling:
[0][1][2][3]             [0][4][8][12]
[4][5][6][7]             [1][5][9][13]
[8][9][10][11]           [2][6][10][14]
[12][13][14][15]         [3][7][11][15]

All threads in column    Threads access
access same bank →       different banks →
SLOW (bank conflict)     FAST (no conflict)
```

### Tiled MMA (Matrix Multiply-Accumulate)

```cpp
typename Kernel_traits::TiledMma tiled_mma;
auto thr_mma = tiled_mma.get_thread_slice(tidx);

// Partition Q for this thread
Tensor tSrQ = thr_mma.partition_fragment_A(sQ);  // Thread's Q fragment

// Partition K for this thread
Tensor tSrK = thr_mma.partition_fragment_B(sK);  // Thread's K fragment

// Compute: each thread computes a part of the result
gemm(tiled_mma, tSrQ, tSrK, acc_s);
```

**What happens:**
- Each thread gets a **fragment** of Q and K
- GPU tensor cores compute the matrix multiply
- Results are accumulated in registers (`acc_s`)

---

## The Flash Attention Algorithm

### Standard Attention (Slow, O(N²) memory)

```python
# Pseudocode
S = Q @ K.T / sqrt(d_k)          # N×N matrix
S = mask(S)                       # Apply masks
P = softmax(S, axis=-1)           # N×N matrix
O = P @ V                         # Output
```

**Problems:**
- Must materialize N×N matrix S
- For N=1024, d=64: S is 4MB (fp32)
- For N=4096: S is 64MB!
- GPU memory bandwidth becomes bottleneck

### Flash Attention (Fast, O(N) memory)

```python
# Pseudocode
O = zeros(N, d)
l = zeros(N)  # Sum of exp
m = ones(N) * -inf  # Max

# Process in blocks
for i in range(0, N, BLOCK_M):  # Query blocks
    Q_i = Q[i:i+BLOCK_M]

    for j in range(0, N, BLOCK_N):  # Key blocks
        K_j = K[j:j+BLOCK_N]
        V_j = V[j:j+BLOCK_N]

        # Compute block attention
        S_ij = Q_i @ K_j.T / sqrt(d_k)  # BLOCK_M × BLOCK_N

        # Online softmax
        m_new = max(m, max(S_ij, axis=-1))
        l = l * exp(m - m_new) + sum(exp(S_ij - m_new), axis=-1)
        O = O * exp(m - m_new)[:, None] + exp(S_ij - m_new) @ V_j
        m = m_new

return O / l[:, None]
```

**Key Differences:**
1. **Never materialize full S**: Only compute BLOCK_M × BLOCK_N at a time
2. **Online statistics**: Update max and sum incrementally
3. **Rescale on-the-fly**: Previous outputs are rescaled when we see new max

**Memory Savings:**
- Standard: O(N²) for S matrix
- Flash: O(BLOCK_M × BLOCK_N) ≈ O(1) since blocks are fixed size

---

## Block-Sparse Optimization

### Sparse Version vs Dense Version

The block-sparse version adds a mask iterator to skip unnecessary blocks.

**File:** [`csrc/block_sparse_attn/src/flash_fwd_kernel.h:649`](../csrc/block_sparse_attn/src/flash_fwd_kernel.h)

```cpp
template<typename Kernel_traits, ...>
inline __device__ void compute_block_attn_1rowblock(
    const Params &params, const int bidb, const int bidh, const int m_block
) {
    // ... (same setup as dense version) ...

    // Create block mask iterator
    fwdIterator<Is_streaming, Is_exact_streaming> blockmask(
        params, binfo, kBlockM, kBlockN, bidb, bidh, m_block, n_block_min, n_block_max
    );

    // Check if this row has any active blocks
    int max_no_larger_idx = blockmask.max_no_larger(n_block_max - 1);
    bool empty_line_flag = (max_no_larger_idx == -1) ||
                          (blockmask.mask_val(max_no_larger_idx) < n_block_min);

    if (empty_line_flag) {
        // No blocks to process - write zeros and return
        // ... write zero output ...
        return;
    }
```

**Key Addition:** Iterator that tells us which blocks are active.

### Block Iteration with Mask

```cpp
    // Outer loop: iterate through iterator indices
    for (int blockmask_iter_idx = max_no_larger_idx;
         blockmask_iter_idx >= 0;
         blockmask_iter_idx--) {

        // Get actual key block index from mask
        int n_block = blockmask.mask_val(blockmask_iter_idx);

        if (n_block < n_block_min) break;  // No more valid blocks

        // Load K and V for this block
        tKgK.data() = tKgK_base.data() + n_block * kBlockN * params.k_row_stride;
        copy(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);

        // Load V
        tVgV.data() = tVgV_base.data() + n_block * kBlockN * params.v_row_stride;
        copy(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);

        // ... (rest same as dense: compute S, softmax, accumulate) ...
    }
```

**What's Different:**
1. **Iterator-based loop** instead of linear `for (n_block = max; n_block >= min; n_block--)`
2. **Skip inactive blocks** entirely - no computation wasted
3. **Non-contiguous access** to K and V using `blockmask.mask_val()`

### Example: Sparse Pattern

```
Query block m=0:
  Active key blocks from mask: [0, 2, 5]  (skips 1, 3, 4)

Loop iterations:
  blockmask_iter_idx=2 → n_block=5  ← Process K[5], V[5]
  blockmask_iter_idx=1 → n_block=2  ← Process K[2], V[2]
  blockmask_iter_idx=0 → n_block=0  ← Process K[0], V[0]

Blocks 1, 3, 4 are never loaded or processed!
```

**Speedup Calculation:**
- If 50% of blocks are inactive (mask=0)
- Sparse version does 50% of the work
- ≈ **2x speedup!**

---

## Complete Execution Flow

### Full Example: ViT with 197 Tokens

**Setup:**
- 197 tokens (196 patches + 1 CLS)
- kBlockM = 128, kBlockN = 128
- Head dimension = 64
- Block-sparse mask with 50% sparsity

**Launch:**
```cpp
num_m_blocks = ceil(197 / 128) = 2
grid = (2, batch_size, num_heads)
```

**Execution for Block (0, 0, 0):** (Query block 0, Batch 0, Head 0)

```
1. Thread block starts
   - m_block = 0 (tokens 0-127)
   - bidb = 0 (first batch item)
   - bidh = 0 (first head)

2. Dispatch
   - head_mask_type[0] = 1 → Use block-sparse
   - Call compute_block_attn_1rowblock<..., false, false>

3. Initialize
   - Load Q[0:128, :] into shared memory
   - acc_o = zeros(128, 64)
   - scores_max = -inf (per row)
   - scores_sum = 0 (per row)

4. Create mask iterator
   - Check which key blocks are active for query block 0
   - Suppose mask says: blocks [0, 1] are active
   - max_no_larger_idx = 1

5. Loop iteration 1: blockmask_iter_idx = 1
   a. n_block = blockmask.mask_val(1) = 1
   b. Load K[128:256, :] and V[128:256, :] into shared memory
   c. Compute S = Q[0:128, :] @ K[128:256, :]^T  → 128×128 scores
   d. softmax_rescale_o (Is_first=true):
      - scores_max = max(S, axis=-1)  → 128 values
      - S = exp2(S - scores_max)
      - scores_sum = sum(S, axis=-1)  → 128 values
   e. P = S (already exp'd and scaled)
   f. acc_o += P @ V[128:256, :]  → accumulate weighted values

6. Loop iteration 2: blockmask_iter_idx = 0
   a. n_block = blockmask.mask_val(0) = 0
   b. Load K[0:128, :] and V[0:128, :] into shared memory
   c. Compute S = Q[0:128, :] @ K[0:128, :]^T
   d. softmax_rescale_o (Is_first=false):
      - scores_max_new = max(scores_max, max(S, axis=-1))
      - scale = exp2(scores_max - scores_max_new)
      - acc_o *= scale  ← Rescale previous accumulator!
      - scores_sum *= scale
      - S = exp2(S - scores_max_new)
      - scores_sum += sum(S, axis=-1)
      - scores_max = scores_max_new
   e. P = S
   f. acc_o += P @ V[0:128, :]

7. Finalize
   - acc_o /= scores_sum  → Normalize
   - lse = log(scores_sum) + scores_max  → Log-sum-exp
   - Convert acc_o from fp32 to fp16
   - Write to global memory O[0:128, :]
   - Write lse[0:128]

8. Done!
```

**Simultaneously:** 95 other CUDA blocks are doing the same for:
- Query block 1 (tokens 128-196)
- Other batch items
- Other heads

### Performance Characteristics

**Memory Accesses:**
- Q: Loaded once per query block
- K, V: Loaded once per (query block, active key block) pair
- Output: Written once per query block

**Compute:**
- GEMM (Q @ K^T): BLOCK_M × BLOCK_N × HEAD_DIM FLOPs
- GEMM (P @ V): BLOCK_M × BLOCK_N × HEAD_DIM FLOPs
- Softmax operations: Minimal compared to GEMMs

**For ViT (197 tokens, 50% sparse):**
- Dense attention: 197² = 38,809 token pairs
- Sparse attention: ~19,404 token pairs (50% saved)
- Each pair requires 2 GEMMs (Q@K^T and P@V)

---

## Summary

### Key Innovations

1. **Tiling:** Process attention in blocks rather than all at once
2. **Online Softmax:** Update statistics incrementally without storing full matrix
3. **Shared Memory:** Leverage fast on-chip memory for blocks
4. **Tensor Cores:** Use specialized hardware for matrix multiplication
5. **Block-Sparse:** Skip unnecessary blocks entirely

### Algorithm Flow

```
For each query block:
  Load Q block once
  For each active key block (from mask):
    Load K, V blocks
    Compute local attention scores
    Update running max and sum
    Rescale previous accumulator
    Accumulate new contribution
  Normalize final output
  Write results
```

### Why It's Fast

1. **Memory bandwidth bound → Compute bound**: Reuse loaded data maximally
2. **O(N²) memory → O(N) memory**: Only store block-sized matrices
3. **Wasted computation eliminated**: Block-sparse skips zero blocks
4. **Hardware acceleration**: Tensor cores for matmul
5. **Fused operations**: Softmax + matmul in one kernel

### Files to Explore Next

- [`flash_blockmask.h`](../csrc/block_sparse_attn/src/flash_blockmask.h) - Iterator implementation
- [`softmax.h`](../csrc/block_sparse_attn/src/softmax.h) - Softmax kernels
- [`kernel_traits.h`](../csrc/block_sparse_attn/src/kernel_traits.h) - Configuration
- [`utils.h`](../csrc/block_sparse_attn/src/utils.h) - Helper functions

---

## Further Reading

### Papers
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691)

### Documentation
- [C++ Techniques Guide](./CPP_TECHNIQUES_GUIDE.md) - CUDA and C++ concepts
- [Implementation Guide](./IMPLEMENTATION_GUIDE.md) - Architecture details
- [Custom Kernel Tutorial](./CUSTOM_KERNEL_TUTORIAL.md) - Hands-on examples

### NVIDIA Resources
- [CUTLASS](https://github.com/NVIDIA/cutlass) - CUDA Templates for Linear Algebra
- [CuTe](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cute) - Tensor abstraction library
- [Tensor Core Programming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
