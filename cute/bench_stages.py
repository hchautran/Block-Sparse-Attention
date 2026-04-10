"""
Benchmark: effect of num_stages (1, 2, 3) on GEMM performance.

Demonstrates how software pipelining hides GMEM latency.

Usage:
    python cute/bench_stages.py
"""

import math
import time
import torch
import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
import cutlass.utils as utils

# ── Config ────────────────────────────────────────────────────────────────────
M, N, K       = 4096, 4096, 4096
BLK_M, BLK_N, BLK_K = 128, 128, 32
MMA_SHAPE_MNK = (16, 8, 16)
ATOM_LAYOUT   = (2, 2, 1)
NUM_THREADS   = 2 * 2 * 1 * 32   # 128
AB_DTYPE      = cutlass.Float16
ACC_DTYPE     = cutlass.Float32

WARMUP_ITERS  = 5
BENCH_ITERS   = 20

# ── SMEM / Copy helpers (shared across all num_stages) ────────────────────────
def make_ab_smem_layout(dtype, is_row_major, blk_mn, blk_k, num_stages):
    major_dim = blk_k if is_row_major else blk_mn
    major_dim = min(major_dim, 64)
    sw = min(int(math.log2(major_dim * dtype.width // 128)), 3)
    if is_row_major:
        outer = cute.make_layout((8, major_dim), stride=(major_dim, 1))
    else:
        outer = cute.make_layout((major_dim, 8), stride=(1, major_dim))
    atom = cute.make_composed_layout(cute.make_swizzle(sw, 3, 3), 0, outer)
    return cute.tile_to_shape(atom, (blk_mn, blk_k, num_stages), (0, 1, 2))


def make_c_smem_layout(blk_m, blk_n):
    # Plain row-major layout — no swizzle needed for epilogue in this benchmark
    return cute.make_layout((blk_m, blk_n), stride=(blk_n, 1))


def make_g2s_tiled_copy(dtype, is_row_major, blk_mn, blk_k, num_threads):
    g2s_atom = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(
            cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
        ),
        dtype,
        num_bits_per_copy=128,
    )
    elems = 128 // dtype.width
    if is_row_major:
        threads_k = blk_k // elems
        thr = cute.make_layout((num_threads // threads_k, threads_k), stride=(threads_k, 1))
        val = cute.make_layout((1, elems))
    else:
        threads_n = blk_mn // elems
        thr = cute.make_layout((threads_n, num_threads // threads_n), stride=(1, threads_n))
        val = cute.make_layout((elems, 1))
    return cute.make_tiled_copy_tv(g2s_atom, thr, val)


def make_epi_tiled_copy(dtype, blk_m, blk_n, num_threads):
    epi_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=128)
    elems = 128 // dtype.width
    threads_n = blk_n // elems
    thr = cute.make_layout((num_threads // threads_n, threads_n), stride=(threads_n, 1))
    val = cute.make_layout((1, elems))
    return cute.make_tiled_copy_tv(epi_atom, thr, val)


# ── Kernels (one per num_stages) ──────────────────────────────────────────────

@cute.kernel
def gemm_kernel_stages1(
    mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
    sA_layout: cute.ComposedLayout, sB_layout: cute.ComposedLayout,
    sC_layout: cute.Layout,
    tiled_copy_A: cute.TiledCopy, tiled_copy_B: cute.TiledCopy,
    tiled_copy_C: cute.TiledCopy,
    tiled_copy_s2r_A: cute.TiledCopy, tiled_copy_s2r_B: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    gA = cute.local_tile(mA, (BLK_M, BLK_K), (bidx, None))
    gB = cute.local_tile(mB, (BLK_N, BLK_K), (bidy, None))
    gC = cute.local_tile(mC, (BLK_M, BLK_N), (bidx, bidy))

    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(AB_DTYPE, sA_layout, byte_alignment=16)
    sB = smem.allocate_tensor(AB_DTYPE, sB_layout, byte_alignment=16)
    sC = cute.make_tensor(cute.recast_ptr(sA.iterator, dtype=AB_DTYPE), sC_layout)

    thr_copy_A = tiled_copy_A.get_slice(tidx)
    tAgA = thr_copy_A.partition_S(gA)
    tAsA = thr_copy_A.partition_D(sA)

    thr_copy_B = tiled_copy_B.get_slice(tidx)
    tBgB = thr_copy_B.partition_S(gB)
    tBsB = thr_copy_B.partition_D(sB)

    thr_mma = tiled_mma.get_slice(tidx)
    tCsA = thr_mma.partition_A(sA)
    tCsB = thr_mma.partition_B(sB)
    tCgC = thr_mma.partition_C(gC)
    tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
    tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
    tCrC = tiled_mma.make_fragment_C(tCgC)
    tCrC.fill(0.0)

    thr_ld_A = tiled_copy_s2r_A.get_slice(tidx)
    tCsA_ld  = thr_ld_A.partition_S(sA)
    tCrA_ld  = thr_ld_A.retile(tCrA)
    thr_ld_B = tiled_copy_s2r_B.get_slice(tidx)
    tCsB_ld  = thr_ld_B.partition_S(sB)
    tCrB_ld  = thr_ld_B.retile(tCrB)

    num_k_tiles  = cute.size(gA, mode=[2])
    num_k_blocks = cute.size(tCrA, mode=[2])

    # ── num_stages = 1: load → WAIT → compute, no overlap ────────────────
    for k_tile in cutlass.range(num_k_tiles):
        tAsA.fill(0); tBsB.fill(0)
        cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile], tAsA[None, None, None, 0])
        cute.copy(tiled_copy_B, tBgB[None, None, None, k_tile], tBsB[None, None, None, 0])
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)   # wait for EVERYTHING before computing
        cute.arch.sync_threads()

        for k_block in cutlass.range_constexpr(num_k_blocks):
            cute.copy(tiled_copy_s2r_A, tCsA_ld[None, None, k_block, 0], tCrA_ld[None, None, k_block])
            cute.copy(tiled_copy_s2r_B, tCsB_ld[None, None, k_block, 0], tCrB_ld[None, None, k_block])
            cute.gemm(tiled_mma, tCrC, tCrA[None, None, k_block], tCrB[None, None, k_block], tCrC)

    cute.arch.sync_threads()
    tCsC = thr_mma.partition_C(sC)
    tCrC_fp16 = cute.make_rmem_tensor(tCsC.shape, AB_DTYPE)
    tCrC_fp16.store(tCrC.load().to(AB_DTYPE))
    cute.autovec_copy(tCrC_fp16, tCsC)
    cute.arch.sync_threads()
    thr_copy_C = tiled_copy_C.get_slice(tidx)
    cute.copy(tiled_copy_C, thr_copy_C.partition_S(sC), thr_copy_C.partition_D(gC))


@cute.kernel
def gemm_kernel_stages2(
    mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
    sA_layout: cute.ComposedLayout, sB_layout: cute.ComposedLayout,
    sC_layout: cute.Layout,
    tiled_copy_A: cute.TiledCopy, tiled_copy_B: cute.TiledCopy,
    tiled_copy_C: cute.TiledCopy,
    tiled_copy_s2r_A: cute.TiledCopy, tiled_copy_s2r_B: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    gA = cute.local_tile(mA, (BLK_M, BLK_K), (bidx, None))
    gB = cute.local_tile(mB, (BLK_N, BLK_K), (bidy, None))
    gC = cute.local_tile(mC, (BLK_M, BLK_N), (bidx, bidy))

    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(AB_DTYPE, sA_layout, byte_alignment=16)
    sB = smem.allocate_tensor(AB_DTYPE, sB_layout, byte_alignment=16)
    sC = cute.make_tensor(cute.recast_ptr(sA.iterator, dtype=AB_DTYPE), sC_layout)

    thr_copy_A = tiled_copy_A.get_slice(tidx)
    tAgA = thr_copy_A.partition_S(gA)
    tAsA = thr_copy_A.partition_D(sA)

    thr_copy_B = tiled_copy_B.get_slice(tidx)
    tBgB = thr_copy_B.partition_S(gB)
    tBsB = thr_copy_B.partition_D(sB)

    thr_mma = tiled_mma.get_slice(tidx)
    tCsA = thr_mma.partition_A(sA)
    tCsB = thr_mma.partition_B(sB)
    tCgC = thr_mma.partition_C(gC)
    tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
    tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
    tCrC = tiled_mma.make_fragment_C(tCgC)
    tCrC.fill(0.0)

    thr_ld_A = tiled_copy_s2r_A.get_slice(tidx)
    tCsA_ld  = thr_ld_A.partition_S(sA)
    tCrA_ld  = thr_ld_A.retile(tCrA)
    thr_ld_B = tiled_copy_s2r_B.get_slice(tidx)
    tCsB_ld  = thr_ld_B.partition_S(sB)
    tCrB_ld  = thr_ld_B.retile(tCrB)

    num_k_tiles  = cute.size(gA, mode=[2])
    num_k_blocks = cute.size(tCrA, mode=[2])

    # ── num_stages = 2: 1 tile prefetch, partial overlap ─────────────────
    # Prologue: prefetch tile 0
    tAsA.fill(0); tBsB.fill(0)
    cute.copy(tiled_copy_A, tAgA[None, None, None, 0], tAsA[None, None, None, 0])
    cute.copy(tiled_copy_B, tBgB[None, None, None, 0], tBsB[None, None, None, 0])
    cute.arch.cp_async_commit_group()

    pipe_read  = cutlass.Int32(0)
    pipe_write = cutlass.Int32(1)

    for k_tile in cutlass.range(num_k_tiles):
        # Issue next tile (overlaps with compute below)
        next_k = k_tile + 1
        if next_k < num_k_tiles:
            cute.copy(tiled_copy_A, tAgA[None, None, None, next_k], tAsA[None, None, None, pipe_write])
            cute.copy(tiled_copy_B, tBgB[None, None, None, next_k], tBsB[None, None, None, pipe_write])
            cute.arch.cp_async_commit_group()

        cute.arch.cp_async_wait_group(0)   # wait for current read slot
        cute.arch.sync_threads()

        for k_block in cutlass.range_constexpr(num_k_blocks):
            cute.copy(tiled_copy_s2r_A, tCsA_ld[None, None, k_block, pipe_read], tCrA_ld[None, None, k_block])
            cute.copy(tiled_copy_s2r_B, tCsB_ld[None, None, k_block, pipe_read], tCrB_ld[None, None, k_block])
            cute.gemm(tiled_mma, tCrC, tCrA[None, None, k_block], tCrB[None, None, k_block], tCrC)

        pipe_read  = (pipe_read  + 1) % 2
        pipe_write = (pipe_write + 1) % 2

    cute.arch.sync_threads()
    tCsC = thr_mma.partition_C(sC)
    tCrC_fp16 = cute.make_rmem_tensor(tCsC.shape, AB_DTYPE)
    tCrC_fp16.store(tCrC.load().to(AB_DTYPE))
    cute.autovec_copy(tCrC_fp16, tCsC)
    cute.arch.sync_threads()
    thr_copy_C = tiled_copy_C.get_slice(tidx)
    cute.copy(tiled_copy_C, thr_copy_C.partition_S(sC), thr_copy_C.partition_D(gC))


@cute.kernel
def gemm_kernel_stages3(
    mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
    sA_layout: cute.ComposedLayout, sB_layout: cute.ComposedLayout,
    sC_layout: cute.Layout,
    tiled_copy_A: cute.TiledCopy, tiled_copy_B: cute.TiledCopy,
    tiled_copy_C: cute.TiledCopy,
    tiled_copy_s2r_A: cute.TiledCopy, tiled_copy_s2r_B: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    gA = cute.local_tile(mA, (BLK_M, BLK_K), (bidx, None))
    gB = cute.local_tile(mB, (BLK_N, BLK_K), (bidy, None))
    gC = cute.local_tile(mC, (BLK_M, BLK_N), (bidx, bidy))

    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(AB_DTYPE, sA_layout, byte_alignment=16)
    sB = smem.allocate_tensor(AB_DTYPE, sB_layout, byte_alignment=16)
    sC = cute.make_tensor(cute.recast_ptr(sA.iterator, dtype=AB_DTYPE), sC_layout)

    thr_copy_A = tiled_copy_A.get_slice(tidx)
    tAgA = thr_copy_A.partition_S(gA)
    tAsA = thr_copy_A.partition_D(sA)

    thr_copy_B = tiled_copy_B.get_slice(tidx)
    tBgB = thr_copy_B.partition_S(gB)
    tBsB = thr_copy_B.partition_D(sB)

    thr_mma = tiled_mma.get_slice(tidx)
    tCsA = thr_mma.partition_A(sA)
    tCsB = thr_mma.partition_B(sB)
    tCgC = thr_mma.partition_C(gC)
    tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
    tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
    tCrC = tiled_mma.make_fragment_C(tCgC)
    tCrC.fill(0.0)

    thr_ld_A = tiled_copy_s2r_A.get_slice(tidx)
    tCsA_ld  = thr_ld_A.partition_S(sA)
    tCrA_ld  = thr_ld_A.retile(tCrA)
    thr_ld_B = tiled_copy_s2r_B.get_slice(tidx)
    tCsB_ld  = thr_ld_B.partition_S(sB)
    tCrB_ld  = thr_ld_B.retile(tCrB)

    num_k_tiles  = cute.size(gA, mode=[2])
    num_k_blocks = cute.size(tCrA, mode=[2])

    # ── num_stages = 3: 2 tiles prefetch, full overlap ───────────────────
    # Prologue: prefetch first 2 tiles
    tAsA.fill(0); tBsB.fill(0)
    cute.arch.sync_threads()
    for s in cutlass.range_constexpr(2):
        cute.copy(tiled_copy_A, tAgA[None, None, None, s], tAsA[None, None, None, s])
        cute.copy(tiled_copy_B, tBgB[None, None, None, s], tBsB[None, None, None, s])
        cute.arch.cp_async_commit_group()

    pipe_read  = cutlass.Int32(0)
    pipe_write = cutlass.Int32(2)

    for k_tile in cutlass.range(num_k_tiles):
        # Issue tile that is 2 ahead (overlap with MMA below)
        next_k = k_tile + 2
        if next_k < num_k_tiles:
            cute.copy(tiled_copy_A, tAgA[None, None, None, next_k], tAsA[None, None, None, pipe_write])
            cute.copy(tiled_copy_B, tBgB[None, None, None, next_k], tBsB[None, None, None, pipe_write])
            cute.arch.cp_async_commit_group()

        # Wait: at most 1 group still in-flight (the one we just issued)
        cute.arch.cp_async_wait_group(1)
        cute.arch.sync_threads()

        for k_block in cutlass.range_constexpr(num_k_blocks):
            cute.copy(tiled_copy_s2r_A, tCsA_ld[None, None, k_block, pipe_read], tCrA_ld[None, None, k_block])
            cute.copy(tiled_copy_s2r_B, tCsB_ld[None, None, k_block, pipe_read], tCrB_ld[None, None, k_block])
            cute.gemm(tiled_mma, tCrC, tCrA[None, None, k_block], tCrB[None, None, k_block], tCrC)

        pipe_read  = (pipe_read  + 1) % 3
        pipe_write = (pipe_write + 1) % 3

    cute.arch.sync_threads()
    tCsC = thr_mma.partition_C(sC)
    tCrC_fp16 = cute.make_rmem_tensor(tCsC.shape, AB_DTYPE)
    tCrC_fp16.store(tCrC.load().to(AB_DTYPE))
    cute.autovec_copy(tCrC_fp16, tCsC)
    cute.arch.sync_threads()
    thr_copy_C = tiled_copy_C.get_slice(tidx)
    cute.copy(tiled_copy_C, thr_copy_C.partition_S(sC), thr_copy_C.partition_D(gC))


# ── Host function (builds all atoms, launches kernel) ────────────────────────

def build_and_run(mA, mB, mC, num_stages, kernel_fn):
    @cute.jit
    def host_fn(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
        sA_layout = make_ab_smem_layout(AB_DTYPE, True, BLK_M, BLK_K, num_stages)
        sB_layout = make_ab_smem_layout(AB_DTYPE, True, BLK_N, BLK_K, num_stages)
        sC_layout = make_c_smem_layout(BLK_M, BLK_N)

        tiled_copy_A    = make_g2s_tiled_copy(AB_DTYPE, True, BLK_M, BLK_K, NUM_THREADS)
        tiled_copy_B    = make_g2s_tiled_copy(AB_DTYPE, True, BLK_N, BLK_K, NUM_THREADS)
        tiled_copy_C    = make_epi_tiled_copy(AB_DTYPE, BLK_M, BLK_N, NUM_THREADS)

        mma_op = cute.nvgpu.warp.MmaF16BF16Op(AB_DTYPE, ACC_DTYPE, MMA_SHAPE_MNK)
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout(ATOM_LAYOUT),
            permutation_mnk=(32, 32, 16),
        )
        s2r_atom_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), AB_DTYPE
        )
        s2r_atom_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), AB_DTYPE
        )
        tiled_copy_s2r_A = cute.make_tiled_copy_A(s2r_atom_A, tiled_mma)
        tiled_copy_s2r_B = cute.make_tiled_copy_B(s2r_atom_B, tiled_mma)

        smem_size = max(
            cute.size_in_bytes(AB_DTYPE, sA_layout) + cute.size_in_bytes(AB_DTYPE, sB_layout),
            cute.size_in_bytes(AB_DTYPE, sC_layout),
        )
        grid = cute.ceil_div(mC.shape, (BLK_M, BLK_N))
        kernel_fn(
            mA, mB, mC,
            sA_layout, sB_layout, sC_layout,
            tiled_copy_A, tiled_copy_B, tiled_copy_C,
            tiled_copy_s2r_A, tiled_copy_s2r_B,
            tiled_mma,
        ).launch(grid=grid, block=[NUM_THREADS, 1, 1], smem=smem_size)

    return cute.compile(host_fn, mA, mB, mC)


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark(fn, mA, mB, mC):
    # warmup
    for _ in range(WARMUP_ITERS):
        fn(mA, mB, mC)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(BENCH_ITERS):
        fn(mA, mB, mC)
    end.record()
    torch.cuda.synchronize()

    ms      = start.elapsed_time(end) / BENCH_ITERS
    tflops  = 2 * M * N * K / (ms * 1e-3) / 1e12
    smem_kb = {1: 16, 2: 32, 3: 48}   # approx A+B smem per stage
    return ms, tflops


def main():
    torch.manual_seed(42)
    a_torch = torch.randn(M, K, dtype=torch.float16).cuda()
    b_torch = torch.randn(N, K, dtype=torch.float16).cuda()
    c_torch = torch.zeros(M, N, dtype=torch.float16).cuda()

    mA = from_dlpack(a_torch)
    mB = from_dlpack(b_torch)
    mC = from_dlpack(c_torch)

    configs = [
        (1, gemm_kernel_stages1, "no overlap: load → stall → compute"),
        (2, gemm_kernel_stages2, "partial:    load N+1 while computing N"),
        (3, gemm_kernel_stages3, "full:       load N+2 while computing N"),
    ]

    print(f"\nGEMM {M}×{N}×{K}, fp16,  tile {BLK_M}×{BLK_N}×{BLK_K},  {NUM_THREADS} threads")
    print("─" * 65)
    print(f"{'num_stages':<12} {'SMEM (A+B)':<14} {'Time (ms)':<12} {'TFLOPS':<10}  description")
    print("─" * 65)

    results = {}
    for num_stages, kernel_fn, desc in configs:
        c_torch.zero_()
        fn = build_and_run(mA, mB, mC, num_stages, kernel_fn)
        ms, tflops = benchmark(fn, mA, mB, mC)
        smem_kb = num_stages * (BLK_M * BLK_K + BLK_N * BLK_K) * 2 // 1024
        results[num_stages] = tflops
        print(f"  {num_stages:<10}  {smem_kb:<4} KB          {ms:<10.3f}  {tflops:<10.2f}  {desc}")

    print("─" * 65)
    print(f"\nSpeedup  stages=2 vs 1: {results[2]/results[1]:.2f}x")
    print(f"Speedup  stages=3 vs 1: {results[3]/results[1]:.2f}x")
    print(f"Speedup  stages=3 vs 2: {results[3]/results[2]:.2f}x")

    # Verify stages=3 result against pytorch
    c_torch.zero_()
    build_and_run(mA, mB, mC, 3, gemm_kernel_stages3)(mA, mB, mC)
    ref = (a_torch.float() @ b_torch.float().t()).half()
    torch.testing.assert_close(c_torch, ref, atol=1e-1, rtol=1e-2)
    print("\nCorrectness check (stages=3): PASSED")


if __name__ == "__main__":
    main()
