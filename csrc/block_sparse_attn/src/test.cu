#include <cstdio>
#include <cmath>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
// #include <cute/arch/mma_80.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

using namespace cute;

template<class T>
__device__ void printObject(T object) {
    if(thread0()) {
        print("\n");
        print(object);
    }
}

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB,
          class TC, class CStride, class CSmemLayout, class TiledMMA,
          class Alpha, class Beta>
__global__ __launch_bounds__(256)
void gemm_tiled_kernel(
    ProblemShape shape_MNK, CtaTiler cta_tiler,
    TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
    TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
    TC      * C, CStride dC, CSmemLayout          , TiledMMA mma,
    Alpha alpha, Beta beta)
{
    using X = Underscore;
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
    
    // Full tensors
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA);
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB);
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC);
    
    // CTA tiles
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});
    
    // SMEM
    __shared__ TA smemA[cosize_v<ASmemLayout>];
    __shared__ TB smemB[cosize_v<BSmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);
    
    // Partition with TiledCopy
    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA);     // (CPY,CPY_M,CPY_K,k)
    // printObject(tAgA);
    Tensor tAsA = thr_copy_a.partition_D(sA);     // (CPY,CPY_M,CPY_K)
    // printObject(tAsA);
    
    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB);     // (CPY,CPY_N,CPY_K,k)
    // printObject(tBgB);
    Tensor tBsB = thr_copy_b.partition_D(sB);     // (CPY,CPY_N,CPY_K)
    // printObject(tBsB);

    // Partition with TiledMMA
    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);        // (MMA,MMA_M,MMA_K)
    Tensor tCsB = thr_mma.partition_B(sB);        // (MMA,MMA_N,MMA_K)
    Tensor tCgC = thr_mma.partition_C(gC);        // (MMA,MMA_M,MMA_N)
    
    // Accumulators
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC);
    
    // Mainloop
    auto K_TILE_MAX = size<3>(tAgA);
    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
        // Use TiledCopy for GMEM -> SMEM
        copy(copy_a, tAgA(_,_,_,k_tile), tAsA);
        copy(copy_b, tBgB(_,_,_,k_tile), tBsB);
        
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();
        
        // Use TiledMMA for compute
        gemm(mma, tCsA, tCsB, tCrC);
        
        __syncthreads();
    }
    
    // Epilogue
    axpby(alpha, tCrC, beta, tCgC);
}

// Host launcher with TiledCopy/TiledMMA (NT layout)
template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_tiled_nt(int m, int n, int k,
                   Alpha alpha,
                   TA const* A, int ldA,
                   TB const* B, int ldB,
                   Beta beta,
                   TC* C, int ldC,
                   cudaStream_t stream = 0)
{
    using namespace cute;
    
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K);
    
    // NT strides
    auto dA = make_stride(Int<1>{}, ldA);
    auto dB = make_stride(Int<1>{}, ldB);
    auto dC = make_stride(Int<1>{}, ldC);
    
    // CTA tile
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<8>{};
    auto cta_tiler = make_shape(bM, bN, bK);
    
    // SMEM layouts with padding to avoid bank conflicts
    auto sA = make_layout(make_shape(bM, bK), make_stride(Int<1>{}, bM + Int<4>{}));
    auto sB = make_layout(make_shape(bN, bK), make_stride(Int<1>{}, bN + Int<4>{}));
    auto sC = make_layout(make_shape(bM, bN));
    
    // TiledCopy: 128-bit vectorized loads
    // Copy_Atom defines the instruction, then we tile with thread and value layouts
    // static_assert(16 % sizeof(TA) == 0, "Vectorized copy requires element size to divide 16 bytes.");
    // static_assert(16 % sizeof(TB) == 0, "Vectorized copy requires element size to divide 16 bytes.");
    // constexpr int kValsPerCopyA = 16 / sizeof(TA);
    // constexpr int kValsPerCopyB = 16 / sizeof(TB);
    // printf("kValsPerCopyA: %d, kValsPerCopyB: %d\n", kValsPerCopyA, kValsPerCopyB);
    // auto val_layout_a = make_layout(make_shape(Int<kValsPerCopyA>{}, Int<1>{}));
    // auto val_layout_b = make_layout(make_shape(Int<kValsPerCopyB>{}, Int<1>{}));
    TiledCopy copy_a = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, TA>{},  // Atom: Copy TAs as if they were uint128_t
        Layout<Shape<_32,_8>>{},                    // Thr layout 32x8 m-major
        Layout<Shape<_4,_1>>{}
    );

    TiledCopy copy_b = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, TB>{},
        Layout<Shape<_32, _8>>{},
        Layout<Shape<_4,_1>>{}
    );
    
    // TiledMMA: 16x16x1 FMA
    TiledMMA mma = make_tiled_mma(
        UniversalFMA<TC, TA, TB>{},     // Scalar FMA
        Layout<Shape<_16, _16, _1>>{}   // 16x16x1 tiling
    );
    
    dim3 dimBlock(size(mma));
    dim3 dimGrid(ceil_div(M, size<0>(cta_tiler)),
                 ceil_div(N, size<1>(cta_tiler)));
    
    gemm_tiled_kernel<<<dimGrid, dimBlock, 0, stream>>>(
        prob_shape, cta_tiler,
        A, dA, sA, copy_a,
        B, dB, sB, copy_b,
        C, dC, sC, mma,
        alpha, beta);
}
int main() {
    const int M = 4096;
    const int N = 4096;
    const int K = 64;
    
    float* A = new float[M * K];
    float* B = new float[N * K];
    float* C = new float[M * N];
    // std::vector<int8_t> A_i8(M * K);
    // std::vector<int8_t> B_i8(N * K);
    
    // Initialize with random values (column-major layout)
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float val = static_cast<float>(rand()) / RAND_MAX;
            A[i + k * M] = val;
            // A_i8[i + k * M] = static_cast<int8_t>((rand() % 127) - 63);
        }
    }
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < K; ++k) {
            float val = static_cast<float>(rand()) / RAND_MAX;
            B[i + k * N] = val;
            // B_i8[i + k * N] = static_cast<int8_t>((rand() % 127) - 63);
        }
    }

    float* d_A;
    float* d_B;
    float* d_C;
    float* d_C_ref;
    // int8_t* d_A_i8;
    // int8_t* d_B_i8;
    int32_t* d_C_i32;

    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, N * K * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMalloc(&d_C_ref, M * N * sizeof(float));
    // cudaMalloc(&d_A_i8, M * K * sizeof(int8_t));
    // cudaMalloc(&d_B_i8, N * K * sizeof(int8_t));
    cudaMalloc(&d_C_i32, M * N * sizeof(int32_t));

    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * K * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_A_i8, A_i8.data(), M * K * sizeof(int8_t), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_B_i8, B_i8.data(), N * K * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, M * N * sizeof(float));
    cudaMemset(d_C_ref, 0, M * N * sizeof(float));
    cudaMemset(d_C_i32, 0, M * N * sizeof(int32_t));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int warmup = 1;
    const int iters = 1;

    // Warmup tensor core version
    for (int i = 0; i < warmup; ++i) {
        gemm_tiled_nt(M, N, K, 1.0f, d_A, M, d_B, N, 0.0 ,d_C, M);
    }
    cudaDeviceSynchronize();

    // Benchmark tensor core version
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        gemm_tiled_nt(M, N, K, 1.0f, d_A, M, d_B, N, 0.0 ,d_C, M);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float cute_tc_ms = 0.0f;
    cudaEventElapsedTime(&cute_tc_ms, start, stop);

    // cuBLAS reference
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha_val = 1.0f;
    const float beta_val = 0.0f;
    
    for (int i = 0; i < warmup; ++i) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    M, N, K, &alpha_val, d_A, M, d_B, N, &beta_val, d_C_ref, M);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    M, N, K, &alpha_val, d_A, M, d_B, N, &beta_val, d_C_ref, M);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float cublas_ms = 0.0f;
    cudaEventElapsedTime(&cublas_ms, start, stop);

    // CuTE INT8 GEMM (accumulate int32)
    // for (int i = 0; i < warmup; ++i) {
        // gemm_tiled_nt(M, N, K, int32_t{1}, d_A_i8, M, d_B_i8, N, int32_t{0}, d_C_i32, M);
    // }
    // cudaDeviceSynchronize();

    // cudaEventRecord(start);
    // for (int i = 0; i < iters; ++i) {
        // gemm_tiled_nt(M, N, K, int32_t{1}, d_A_i8, M, d_B_i8, N, int32_t{0}, d_C_i32, M);
    // }
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // float cute_i8_ms = 0.0f;
    // cudaEventElapsedTime(&cute_i8_ms, start, stop);

    const int32_t alpha_i32 = 1;
    const int32_t beta_i32 = 0;
    // for (int i = 0; i < warmup; ++i) {
    //     cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
    //                 M, N, K,
    //                 &alpha_i32,
    //                 d_A_i8, CUDA_R_8I, M,
    //                 d_B_i8, CUDA_R_8I, N,
    //                 &beta_i32,
    //                 d_C_i32, CUDA_R_32I, M,
    //                 CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    // }
    // cudaDeviceSynchronize();

    // cudaEventRecord(start);
    // for (int i = 0; i < iters; ++i) {
        // cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
    //                 M, N, K,
    //                 &alpha_i32,
    //                 d_A_i8, CUDA_R_8I, M,
    //                 d_B_i8, CUDA_R_8I, N,
    //                 &beta_i32,
    //                 d_C_i32, CUDA_R_32I, M,
    //                 CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    // }
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // float cublas_i8_ms = 0.0f;
    // cudaEventElapsedTime(&cublas_i8_ms, start, stop);

    // Validation
    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    std::vector<float> C_ref(M * N);
    cudaMemcpy(C_ref.data(), d_C_ref, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float diff = std::fabs(C[i] - C_ref[i]);
        max_diff = std::max(max_diff, diff);
        avg_diff += diff;
    }
    avg_diff /= (M * N);
    
    double tc_gflops = (2.0 * M * N * K * 1e-9) / ((cute_tc_ms / iters) * 1e-3);
    double cublas_gflops = (2.0 * M * N * K * 1e-9) / ((cublas_ms / iters) * 1e-3);
    // double cute_i8_tops = (2.0 * M * N * K * 1e-12) / ((cute_i8_ms / iters) * 1e-3);
    // double cublas_i8_tops = (2.0 * M * N * K * 1e-12) / ((cublas_i8_ms / iters) * 1e-3);
    
    printf("=== Validation Results ===\n");
    printf("Max diff vs cuBLAS: %e\n", max_diff);
    printf("Avg diff vs cuBLAS: %e\n", avg_diff);
    printf("Status: %s\n\n", max_diff < 1e-2 ? "✓ PASS" : "✗ FAIL");  // TF32 has lower precision
    
    printf("=== Performance Results ===\n");
    printf("Problem size: %dx%dx%d\n", M, N, K);
    printf("CuTe Tensor Core: %.3f ms (%.1f GFLOPS)\n", cute_tc_ms / iters, tc_gflops);
    printf("cuBLAS SGEMM:     %.3f ms (%.1f GFLOPS)\n", cublas_ms / iters, cublas_gflops);
    // printf("CuTE INT8:        %.3f ms (%.2f TOPS)\n", cute_i8_ms / iters, cute_i8_tops);
    // printf("cuBLAS INT8:      %.3f ms (%.2f TOPS)\n", cublas_i8_ms / iters, cublas_i8_tops);
    printf("Efficiency:       %.1f%% of cuBLAS\n", (tc_gflops / cublas_gflops) * 100);
    printf("Speedup vs naive: %.1fx\n", 11226.1 / tc_gflops);  // vs your 59% baseline
    
    cudaDeviceSynchronize();
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_ref);
    // cudaFree(d_A_i8);
    // cudaFree(d_B_i8);
    cudaFree(d_C_i32);
    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}
