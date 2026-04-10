# CuTe DSL Tensor Core GEMM Tutorial

A step-by-step guide to implementing `C = A @ B` using CuTe DSL with Ampere (SM80) tensor cores.

- **A**: `(M, K)` row-major — K is contiguous
- **B**: `(N, K)` row-major — K is contiguous
- **C**: `(M, N)` row-major — N is contiguous
- **Target**: Ampere SM80, `mma.sync.m16n8k16` warp-level tensor core

---

## The Full Pipeline

```
A (GMEM) ──cp.async──► sA (SMEM) ──ldmatrix──► tCrA (RMEM) ─┐
                                                               ├──► cute.gemm ──► tCrC (RMEM) ──► sC (SMEM) ──► C (GMEM)
B (GMEM) ──cp.async──► sB (SMEM) ──ldmatrix──► tCrB (RMEM) ─┘
```

Each arrow is a distinct copy operation with its own atom and layout.

---

## Part 0 — Config

```python
import math
import torch
import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
import cutlass.utils as utils

M, N, K = 1024, 1024, 256

BLK_M, BLK_N, BLK_K = 128, 128, 32   # tile size per CTA
NUM_STAGES           = 3               # SMEM pipeline depth
MMA_SHAPE_MNK        = (16, 8, 16)    # one warp MMA instruction shape
ATOM_LAYOUT_MNK      = (2, 2, 1)      # warp replication across M, N, K
NUM_THREADS          = 2 * 2 * 1 * 32 # = 128

AB_DTYPE  = cutlass.Float16
ACC_DTYPE = cutlass.Float32

A = from_dlpack(torch.randn(M, K, dtype=torch.float16).cuda())  # (M, K) row-major
B = from_dlpack(torch.randn(N, K, dtype=torch.float16).cuda())  # (N, K) row-major
C = from_dlpack(torch.zeros(M, N, dtype=torch.float16).cuda())  # (M, N) row-major
```

---

## Part 1 — SMEM Layout for A and B

### What is a CuTe Layout?

A layout is `(shape, stride)`. It maps a multi-dimensional index to a flat memory offset.

```
A = (M, K):(K, 1)    row-major:  A[m, k] = base + m*K + k*1
B = (K, N):(1, N)    col-major:  B[k, n] = base + k*1 + n*N
```

### Why a Special SMEM Layout?

SMEM needs two things baked into its layout:

**1. A base tile shape that matches the copy atom (128-bit)**

For fp16 with 128-bit async copy → 8 elements per load.
We tile 8 rows × BLK_K cols as the layout atom:

```
layout_atom_outer = (8, BLK_K):(BLK_K, 1)    8 rows, BLK_K cols, K-contiguous
```

**2. Swizzle to eliminate ldmatrix bank conflicts**

`ldmatrix` issues one warp-wide 128-byte load: 32 threads each fetch 4 bytes from 32
different SMEM banks. Without swizzling, multiple threads can hit the same bank and stall.

```
Without swizzle:                     With Swizzle(2, 3, 3):
  Thread  0 → bank 0  ← conflict      Thread  0 → bank  0
  Thread  1 → bank 0  ← conflict      Thread  1 → bank  1
  Thread  2 → bank 0  ← conflict      Thread  2 → bank  2
  ...                                  ...  (all different banks)
```

`Swizzle(B, M, S)` XORs bits `[M+S : M+S-B]` of the column address into bits `[M : M-B]`
of the row address, spreading rows across banks.

```
For A (row-major, fp16, BLK_K=32, 128-bit copy):

  major_dim    = BLK_K = 32
  swizzle_bits = min(log2(32 * 16 / 128), 3) = min(log2(4), 3) = 2

  atom_outer = (8, 32):(32, 1)           8 rows × 32 cols, row-major
  atom       = ComposedLayout(Swizzle(2, 3, 3), 0, atom_outer)

  sA_layout  = tile_to_shape(atom, (BLK_M, BLK_K, NUM_STAGES), (0, 1, 2))
                                    ──────  ─────  ───────────
                                     M dim   K dim   pipeline
```

```python
def make_ab_smem_layout(dtype, is_row_major, shape_mnk):
    # shape_mnk = (BLK_MN, BLK_K, NUM_STAGES)
    major_dim = shape_mnk[1] if is_row_major else shape_mnk[0]
    major_dim = min(major_dim, 64)
    sw = min(int(math.log2(major_dim * dtype.width // 128)), 3)

    if is_row_major:
        outer = cute.make_layout((8, major_dim), stride=(major_dim, 1))
    else:
        outer = cute.make_layout((major_dim, 8), stride=(1, major_dim))

    atom = cute.make_composed_layout(cute.make_swizzle(sw, 3, 3), 0, outer)
    return cute.tile_to_shape(atom, shape_mnk, (0, 1, 2))

sA_layout = make_ab_smem_layout(AB_DTYPE, True,  (BLK_M, BLK_K, NUM_STAGES))
sB_layout = make_ab_smem_layout(AB_DTYPE, True,  (BLK_N, BLK_K, NUM_STAGES))
```

**SMEM shape summary:**

```
sA: (128, 32, 3)  →  128 × 32 × 3 × 2B = 24 KB
sB: (128, 32, 3)  →  128 × 32 × 3 × 2B = 24 KB
                                          48 KB total for A + B
```

> **Rule**: The contiguous dimension of the SMEM layout must match the contiguous
> dimension of the global tensor. This maximizes coalescing in the GMEM→SMEM copy.

---

## Part 2 — GMEM → SMEM: Asynchronous Copy

### Step A: Define the Copy Atom

The atom is the smallest unit one thread copies: **128 bits = 8 fp16 elements**.
This maps to `cp.async.ca.shared.global [dst], [src], 16` in PTX.

```python
g2s_atom = cute.make_copy_atom(
    cute.nvgpu.cpasync.CopyG2SOp(
        cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL  # bypass L1, fill L2
    ),
    AB_DTYPE,
    num_bits_per_copy=128,
)
```

### Step B: Tile the Atom Across Threads

For row-major A (K contiguous), each thread copies 8 contiguous K elements:

```
BLK_M=128, BLK_K=32, 128 threads, 8 elems/thread:

  threads_along_K = BLK_K / 8 = 4
  threads_along_M = 128 / 4   = 32

  thread_layout = (32, 4):(4, 1)   32 thread-rows × 4 thread-cols
  value_layout  = (1, 8)            1 row × 8 cols per thread
```

```python
def make_g2s_tiled_copy(is_row_major, blk_mn):
    elems = 128 // AB_DTYPE.width        # 8 fp16 per 128-bit copy
    if is_row_major:
        threads_along_k = BLK_K // elems
        thr_layout = cute.make_layout(
            (NUM_THREADS // threads_along_k, threads_along_k),
            stride=(threads_along_k, 1),
        )
        val_layout = cute.make_layout((1, elems))
    else:
        threads_along_n = blk_mn // elems
        thr_layout = cute.make_layout(
            (threads_along_n, NUM_THREADS // threads_along_n),
            stride=(1, threads_along_n),
        )
        val_layout = cute.make_layout((elems, 1))
    return cute.make_tiled_copy_tv(g2s_atom, thr_layout, val_layout)

tiled_copy_A = make_g2s_tiled_copy(True, BLK_M)
tiled_copy_B = make_g2s_tiled_copy(True, BLK_N)
```

### Step C: Inside the Kernel

```python
thr_copy_A = tiled_copy_A.get_slice(tidx)

# Partition global and shared tensors for this thread
tAgA = thr_copy_A.partition_S(gA)   # source: global A  (CPY, CPY_M, CPY_K, k_tiles)
tAsA = thr_copy_A.partition_D(sA)   # dest:   shared A  (CPY, CPY_M, CPY_K, stages)

# Prologue: prefetch first (NUM_STAGES - 1) tiles before the main loop
for stage in range(NUM_STAGES - 1):
    cute.copy(tiled_copy_A, tAgA[None, None, None, stage], tAsA[None, None, None, stage])
    cute.copy(tiled_copy_B, tBgB[None, None, None, stage], tBsB[None, None, None, stage])
    cute.arch.cp_async_commit_group()   # mark this batch as a commit group

# In the main loop, wait before computing:
cute.arch.cp_async_wait_group(NUM_STAGES - 2)  # at most 1 group still in-flight
cute.arch.sync_threads()
```

> **Why `wait_group(NUM_STAGES - 2)`?**
> With 3 stages, `wait_group(1)` means "at most 1 group still in-flight".
> The group we are about to read is guaranteed done, while the next group can
> still be loading. This is the key overlap that hides memory latency.

---

## Part 3 — SMEM → Registers: LdMatrix

### Why LdMatrix?

Tensor cores require operands in a specific register arrangement spread across all 32
threads of a warp. Computing the correct index per-thread manually is slow and error-prone.

`ldmatrix` is a **single warp-cooperative PTX instruction** that:
- Each of the 32 threads provides one SMEM pointer
- Together the warp loads an 8×8 matrix of 16-bit values
- Each thread ends up holding its correct piece for the subsequent MMA instruction

### Setting Up LdMatrix

```python
# A is row-major → transpose=False (K contiguous matches tensor core A operand)
# B is row-major → transpose=False (K contiguous matches tensor core B operand)
# For col-major B: transpose=True (N contiguous → needs transposing for tensor core)

s2r_atom_A = cute.make_copy_atom(
    cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
    AB_DTYPE,
)
s2r_atom_B = cute.make_copy_atom(
    cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
    AB_DTYPE,
)
```

`num_matrices=4`: loads 4 × (8×8×16b) tiles = 512 bytes per warp = full m16n8k16 A operand.

### Why `make_tiled_copy_A` Instead of `make_tiled_copy_tv`?

For ldmatrix the thread layout is **not** free to choose. It must exactly match what
the MMA instruction expects to receive in registers. CuTe derives it automatically:

```python
tiled_copy_s2r_A = cute.make_tiled_copy_A(s2r_atom_A, tiled_mma)  # layout derived from MMA
tiled_copy_s2r_B = cute.make_tiled_copy_B(s2r_atom_B, tiled_mma)
```

### Inside the Kernel — Load Registers

```python
thr_ldmat_A = tiled_copy_s2r_A.get_slice(tidx)
tCsA_view   = thr_ldmat_A.partition_S(sA)   # SMEM source  (LDMAT, LDMAT_M, LDMAT_K, stages)
tCrA_view   = thr_ldmat_A.retile(tCrA)      # Reg dest — reinterpret fragment layout

# retile() re-interprets the existing register buffer in ldmatrix's shape.
# It does NOT allocate new registers. Both tCrA and tCrA_view point to the same registers.

for k_block in range(num_k_blocks):
    cute.copy(tiled_copy_s2r_A, tCsA_view[None, None, k_block, stage], tCrA_view[None, None, k_block])
    cute.copy(tiled_copy_s2r_B, tCsB_view[None, None, k_block, stage], tCrB_view[None, None, k_block])
    cute.gemm(tiled_mma, tCrC, tCrA[None, None, k_block], tCrB[None, None, k_block], tCrC)
```

> **k-tile vs k-block**
>
> A **k-tile** fills one SMEM stage: size = BLK_K (e.g. 32 elements).
> A **k-block** is the MMA instruction's K footprint: size = MMA_K (e.g. 16 elements).
> One k-tile = BLK_K / MMA_K = 32 / 16 = **2 k-blocks**.

---

## Part 4 — Tiled MMA

### Building the Tiled MMA

```python
# Single Ampere warp MMA instruction: m16n8k16 fp16 → fp32
mma_op = cute.nvgpu.warp.MmaF16BF16Op(
    ab_dtype=cutlass.Float16,
    acc_dtype=cutlass.Float32,
    shape_mnk=(16, 8, 16),   # one warp computes 16×8 output using 16 K-elements
)

# Replicate across warps: 2×2×1 atom layout → 4 warps = 128 threads
tiled_mma = cute.make_tiled_mma(
    mma_op,
    cute.make_layout((2, 2, 1)),    # 2 warps in M, 2 warps in N, 1 in K
    permutation_mnk=(32, 32, 16),   # each CTA covers a 32×32×16 MNK region
)
```

### Fragment Allocation

```python
thr_mma = tiled_mma.get_slice(tidx)

tCsA = thr_mma.partition_A(sA)                          # (MMA_ATOM, MMA_M, MMA_K, stages)
tCsB = thr_mma.partition_B(sB)                          # (MMA_ATOM, MMA_N, MMA_K, stages)
tCgC = thr_mma.partition_C(gC)                          # (MMA_ATOM, MMA_M, MMA_N)

tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])  # register buffer for A
tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])  # register buffer for B
tCrC = tiled_mma.make_fragment_C(tCgC)                        # accumulator (fp32)
tCrC.fill(0.0)
```

> `make_fragment_A/B` allocates registers in the **exact layout the tensor core hardware
> expects**. This is why `retile` works: ldmatrix and make_fragment_A produce the same
> register arrangement, just with a different shape view.

---

## Part 5 — Store C: RMEM → SMEM → GMEM

### Why Stage Through SMEM?

After MMA, the accumulator elements are **scattered** across M×N space following the MMA
thread partition. Writing directly to GMEM gives non-coalesced, low-bandwidth stores.

Staging through SMEM with a different copy pattern lets all threads write contiguous
128-bit chunks to GMEM:

```
MMA partition (scattered)      SMEM C (repacked)      GMEM C (coalesced)
  tCrC: (MMA, MMA_M, MMA_N) ──autovec_copy──► sC ──tiled_copy_C──► gC
```

### SMEM Layout for C

C reuses A's SMEM allocation (A and C never coexist):

```python
def make_c_smem_layout(dtype, is_row_major):
    major_dim = BLK_N if is_row_major else BLK_M
    sw = min(int(math.log2(major_dim * dtype.width // 128)), 3)
    if is_row_major:
        outer = cute.make_layout((8, major_dim), stride=(major_dim, 1))
    else:
        outer = cute.make_layout((major_dim, 8), stride=(1, major_dim))
    atom = cute.make_composed_layout(cute.make_swizzle(sw, 3, 4), 0, outer)
    return cute.tile_to_shape(atom, (BLK_M, BLK_N), (0, 1))

sC_layout = make_c_smem_layout(AB_DTYPE, is_row_major=True)
sC = cute.make_tensor(cute.recast_ptr(sA.iterator, dtype=AB_DTYPE), sC_layout)
```

### Epilogue Steps

```python
# ── Step 1: RMEM → SMEM ─────────────────────────────────────────────────────
cute.arch.sync_threads()                 # ensure mainloop done before overwriting sA memory

tCsC = thr_mma.partition_C(sC)          # SMEM C view using MMA partition
tCrC_fp16 = cute.make_rmem_tensor(tCsC.shape, AB_DTYPE)
tCrC_fp16.store(tCrC.load().to(AB_DTYPE))   # fp32 → fp16 conversion in registers
cute.autovec_copy(tCrC_fp16, tCsC)           # auto-vectorized scatter to SMEM

cute.arch.sync_threads()                     # all threads done writing SMEM

# ── Step 2: SMEM → GMEM ─────────────────────────────────────────────────────
epi_atom = cute.make_copy_atom(
    cute.nvgpu.CopyUniversalOp(),
    AB_DTYPE,
    num_bits_per_copy=128,
)
tiled_copy_C = make_g2s_tiled_copy(is_row_major=True, blk_mn=BLK_N)

thr_copy_C = tiled_copy_C.get_slice(tidx)
tCsC_epi   = thr_copy_C.partition_S(sC)    # source: SMEM C
tCgC_epi   = thr_copy_C.partition_D(gC)    # dest:   GMEM C

cute.copy(tiled_copy_C, tCsC_epi, tCgC_epi)   # coalesced 128-bit stores
```

---

## Part 6 — 3-Stage Pipeline Mainloop

### Timing Diagram

```
K-tile:     0          1          2          3          4
            ├──load────┤──load────┤──load────┤──load────┤──load────►
                       ├──compute─┤──compute─┤──compute─┤──compute─►
                       ^
                  wait_group(1) ensures tile 0 is ready before compute
```

### Code

```python
# Prologue: fill pipeline with first (NUM_STAGES - 1) tiles
tAsA.fill(0); tBsB.fill(0)
cute.arch.sync_threads()

for stage in range(NUM_STAGES - 1):
    cute.copy(tiled_copy_A, tAgA[None, None, None, stage], tAsA[None, None, None, stage])
    cute.copy(tiled_copy_B, tBgB[None, None, None, stage], tBsB[None, None, None, stage])
    cute.arch.cp_async_commit_group()

tCrC.fill(0.0)
smem_pipe_read  = 0
smem_pipe_write = NUM_STAGES - 1
num_k_tiles = cute.size(gA, mode=[2])

for k_tile in cutlass.range(num_k_tiles):

    # Wait for the current read stage to arrive
    cute.arch.cp_async_wait_group(NUM_STAGES - 2)
    cute.arch.sync_threads()

    # Issue next global load (overlaps with MMA below)
    next_k = k_tile + NUM_STAGES - 1
    if next_k < num_k_tiles:
        cute.copy(tiled_copy_A, tAgA[None, None, None, next_k], tAsA[None, None, None, smem_pipe_write])
        cute.copy(tiled_copy_B, tBgB[None, None, None, next_k], tBsB[None, None, None, smem_pipe_write])
        cute.arch.cp_async_commit_group()

    # LdMatrix + MMA over all k-blocks in this tile
    num_k_blocks = cute.size(tCrA, mode=[2])   # BLK_K / MMA_K = 2
    for k_block in cutlass.range_constexpr(num_k_blocks):
        cute.copy(tiled_copy_s2r_A, tCsA_view[None, None, k_block, smem_pipe_read], tCrA_view[None, None, k_block])
        cute.copy(tiled_copy_s2r_B, tCsB_view[None, None, k_block, smem_pipe_read], tCrB_view[None, None, k_block])
        cute.gemm(tiled_mma, tCrC, tCrA[None, None, k_block], tCrB[None, None, k_block], tCrC)

    smem_pipe_read  = (smem_pipe_read  + 1) % NUM_STAGES
    smem_pipe_write = (smem_pipe_write + 1) % NUM_STAGES
```

---

## Part 7 — Naming Convention

CuTe uses a systematic prefix convention. Reading a tensor name tells you exactly what it is:

| Prefix | Memory location | Example |
|--------|-----------------|---------|
| `m`    | Global (full matrix) | `mA`, `mB`, `mC` |
| `g`    | Global (this CTA's tile) | `gA`, `gB`, `gC` |
| `s`    | Shared memory | `sA`, `sB`, `sC` |
| `r`    | Register | `tCrA`, `tCrB`, `tCrC` |

Second letter encodes which operation partitioned it:

| Second letter | Partitioned by |
|---------------|----------------|
| `A`           | tiled_copy_A (GMEM→SMEM copy for A) |
| `B`           | tiled_copy_B (GMEM→SMEM copy for B) |
| `C`           | tiled_mma (MMA operation) |

So:
- `tAsA` = thread-A-copy's view of shared-A (source and dest of the A async copy)
- `tCsA` = thread-C-MMA's view of shared-A (A operand as seen by the MMA partition)
- `tCrC` = thread-C-MMA's register-C (the accumulator)

---

## Summary: What Each Piece Controls

| Component | What it controls |
|-----------|-----------------|
| `sA_layout` swizzle | Bank conflict avoidance for ldmatrix reads |
| `sA_layout` pipeline dim | How many K-tiles fit in SMEM (= NUM_STAGES) |
| `tiled_copy_A` thread layout | Coalescing pattern for GMEM→SMEM loads |
| `LdMatrix8x8x16bOp(num_matrices=4)` | Loads full MMA operand in one instruction |
| `LdMatrix8x8x16bOp(transpose)` | Whether to transpose B during the load |
| `make_tiled_copy_A(atom, tiled_mma)` | Ensures ldmatrix output matches MMA input layout |
| `ATOM_LAYOUT_MNK` | How many warps collaborate on one CTA tile |
| `retile` vs `partition_D` | Reinterpret existing register buffer vs allocate new one |
| `autovec_copy` in epilogue | Scatter accumulator from MMA layout → contiguous SMEM |
| SMEM staging for C | Enables coalesced 128-bit stores to GMEM |
