#include <cutlass/numeric_types.h>
#include "kernel_traits.h"
#include "flash_params.h"
#include "flash_fwd_kernel_inference.h"
#include "cute/algorithm/copy.hpp"
#include <cute/tensor.hpp>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace cute;

template<typename Kernel_traits >
__global__ void dummy_kernel(cutlass::half_t* params) {
    int tidx = threadIdx.x;

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor gQ = make_tensor(make_gmem_ptr(params),
            Shape<Int<Kernel_traits::kBlockM>, Int<64>>{},
            make_stride(1024, _1{}));  // [kBlockM, kHeadDim]

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);  // per-thread Q gmem tile
    if (cute::thread0()) {
        printf("======\n");
        print(Kernel_traits::kBlockM );
        printf("\n");
        printf("======\n");
        print(gQ);
        printf("\n");
        printf("======\n");
        print(tQgQ);
        printf("\n");
    }
}

__global__ void elementwise_add(const cutlass::half_t* a,
                                const cutlass::half_t* b,
                                cutlass::half_t* out,
                                int total_elems) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elems) {
        out[idx] = a[idx] + b[idx];
    }
}

template<typename Kernel_traits>
void run_dummy_kernel(cutlass::half_t* params) {
    // SAM-B uses head_dim=64
    // Block config: 128x128 (optimized for A100/H100)
   dummy_kernel<Kernel_traits><<<cute::ceil_div(1024, 128), Kernel_traits::kNThreads, 0>>>(params);
   cudaError_t launch_err = cudaGetLastError();
   if (launch_err != cudaSuccess) {
       printf("Kernel launch error: %s\n", cudaGetErrorString(launch_err));
   }
}

template<typename Kernel_traits, bool Is_even_MN, bool Is_even_K>
__global__ void flash_fwd_test_kernel(FLASH_NAMESPACE::Flash_fwd_params params) {
    FLASH_NAMESPACE::compute_attn<Kernel_traits, Is_even_MN, Is_even_K>(params);
}

void run_block_attn_sam_smoke() {
    using KernelTraits = Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass::half_t>;
    constexpr int kBlockM = KernelTraits::kBlockM;
    constexpr int kBlockN = KernelTraits::kBlockN;
    constexpr int kHeadDim = KernelTraits::kHeadDim;

    const int b = 1;
    const int h = 16;
    const int seqlen_q = kBlockM * 32;
    const int seqlen_k = kBlockN * 32;
    const int d = kHeadDim;
    const int total_elems = b * seqlen_q * h * d;

    std::vector<cutlass::half_t> host_q(total_elems);
    std::vector<cutlass::half_t> host_k(total_elems);
    std::vector<cutlass::half_t> host_v(total_elems);
    std::vector<cutlass::half_t> host_pos( b * h * seqlen_q * seqlen_q);
    std::vector<int> host_blockmask(b * h * seqlen_q * seqlen_q, 1);
    std::vector<int> head_mask_type(h, 1);
    for (int i = 0; i < total_elems; ++i) {
        host_q[i] = static_cast<cutlass::half_t>(0.01f * (i % 7));
        host_k[i] = static_cast<cutlass::half_t>(0.02f * (i % 5));
        host_v[i] = static_cast<cutlass::half_t>(0.03f * (i % 3));
    }

    for (int i = 0; i <  b * h * seqlen_q * seqlen_q; ++i) {
        host_pos[i] = static_cast<cutlass::half_t>(i);
    }
    // {
    //     const int tile = 128;
    //     const int head_stride = 2 * seqlen_q * seqlen_q;
    //     const int base = 0 * head_stride;
    //     // printf("first 32x32 tile (b=0,h=2):\n");
    //     // for (int r = 0; r < tile; ++r) {
    //     //     const int row_offset = base + r * seqlen_q;
    //     //     for (int c = 0; c < tile; ++c) {
    //     //         printf("%3.0f ", static_cast<float>(host_pos[row_offset + c]));
    //     //     }
    //     //     printf("\n");
    //     // }
    //     const int last_row = seqlen_q - tile;
    //     const int last_col = seqlen_q - tile;
    //     printf("last 128x128 tile (b=0,h=2):\n");
    //     for (int r = 0; r < tile; ++r) {
    //         const int row_offset = base + (last_row + r) * seqlen_q + last_col;
    //         for (int c = 0; c < tile; ++c) {
    //             printf("%3.0f ", static_cast<float>(host_pos[row_offset + c]));
    //         }
    //         printf("\n");
    //     }
    // }


    cutlass::half_t *device_q = nullptr, *device_k = nullptr, *device_v = nullptr, *device_o = nullptr, * device_pos= nullptr;
    float *device_lse = nullptr;
    int *device_blockmask = nullptr, *device_head_mask_type = nullptr;
    cudaMalloc(&device_q, total_elems * sizeof(cutlass::half_t));
    cudaMalloc(&device_k, total_elems * sizeof(cutlass::half_t));
    cudaMalloc(&device_v, total_elems * sizeof(cutlass::half_t));
    cudaMalloc(&device_o, total_elems * sizeof(cutlass::half_t));
    cudaMalloc(&device_lse, b * h * seqlen_q * sizeof(float));
    cudaMalloc(&device_pos, b * h * seqlen_q * seqlen_q * sizeof(float));

    cudaMemcpy(device_q, host_q.data(), total_elems * sizeof(cutlass::half_t), cudaMemcpyHostToDevice);
    cudaMemcpy(device_k, host_k.data(), total_elems * sizeof(cutlass::half_t), cudaMemcpyHostToDevice);
    cudaMemcpy(device_v, host_v.data(), total_elems * sizeof(cutlass::half_t), cudaMemcpyHostToDevice);
    cudaMemcpy(device_pos, host_pos.data(), b * h * seqlen_q * seqlen_q * sizeof(cutlass::half_t), cudaMemcpyHostToDevice);

    cudaMalloc(&device_blockmask, host_blockmask.size() * sizeof(int));
    cudaMalloc(&device_head_mask_type, head_mask_type.size() * sizeof(int));
    cudaMemcpy(device_blockmask, host_blockmask.data(), host_blockmask.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_head_mask_type, head_mask_type.data(), head_mask_type.size() * sizeof(int), cudaMemcpyHostToDevice);

    FLASH_NAMESPACE::Flash_fwd_params params{};
    params.q_ptr = device_q;
    params.k_ptr = device_k;
    params.v_ptr = device_v;
    params.o_ptr = device_o;
    params.softmax_lse_ptr = device_lse;
    params.q_row_stride = h * d;
    params.k_row_stride = h * d;
    params.v_row_stride = h * d;
    params.o_row_stride = h * d;
    params.q_head_stride = d;
    params.k_head_stride = d;
    params.v_head_stride = d;
    params.o_head_stride = d;
    params.q_batch_stride = seqlen_q * h * d;
    params.k_batch_stride = seqlen_k * h * d;
    params.v_batch_stride = seqlen_k * h * d;
    params.o_batch_stride = seqlen_q * h * d;
    params.b = b;
    params.h = h;
    params.h_k = h;
    params.h_h_k_ratio = 1;
    params.d = d;
    params.d_rounded = d;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q;
    params.seqlen_k_rounded = seqlen_k;
    params.total_q = b * seqlen_q;
    params.cu_seqlens_q = nullptr;
    params.cu_seqlens_k = nullptr;
    params.seqused_k = nullptr;
    params.knew_ptr = nullptr;
    params.seqlen_knew = 0;
    params.blockmask = device_blockmask;
    params.head_mask_type = device_head_mask_type;
    params.m_block_dim = kBlockM;
    params.n_block_dim = kBlockN;
    params.num_blocksparse_heads = 1;
    params.pos_ptr = device_pos;
    params.pos_batch_stride = 16*4096*4096;
    params.pos_head_stride = 4096*4096;
    params.pos_row_stride = 4096;
    params.pos_col_stride = 1;
    const float scale = 1.0f / sqrtf(static_cast<float>(d));
    params.scale_softmax = scale;
    params.scale_softmax_log2 = scale * 1.4426950408889634f;
    params.is_bf16 = false;
    params.is_seqlens_k_cumulative = true;

    constexpr bool kEvenMN = true;
    constexpr bool kEvenK = true;
    auto kernel = &flash_fwd_test_kernel<KernelTraits, kEvenMN, kEvenK>;
    const size_t smem_size = KernelTraits::kSmemSize;
    if (smem_size >= 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }
    dim3 grid(1, b, h);
    kernel<<<grid, KernelTraits::kNThreads, smem_size>>>(params);
    //todo: help me return the attention map so that I can check whether positional bias adding is correct 
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        printf("Block attn kernel launch error: %s\n", cudaGetErrorString(launch_err));
    }
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        printf("Block attn kernel sync error: %s\n", cudaGetErrorString(sync_err));
    }

    std::vector<cutlass::half_t> host_o(total_elems);
    cudaMemcpy(host_o.data(), device_o, total_elems * sizeof(cutlass::half_t), cudaMemcpyDeviceToHost);
    printf("Block attn O[0]=%f\n", static_cast<float>(host_o[4096]));

    cudaFree(device_q);
    cudaFree(device_k);
    cudaFree(device_v);
    cudaFree(device_o);
    cudaFree(device_lse);
    cudaFree(device_blockmask);
    cudaFree(device_head_mask_type);
}

int main() {
    // const int rows = 4096;
    // const int cols = 1024; 
    // const int total = rows * cols;
    // std::vector<cutlass::half_t> host_a(total);
    // std::vector<cutlass::half_t> host_b(total);
    // for (int i = 0; i < total; ++i) {
        // host_a[i] = static_cast<cutlass::half_t>(i);
        // host_b[i] = static_cast<cutlass::half_t>(1);
    // }
    // cutlass::half_t *device_a = nullptr, *device_b = nullptr, *device_out = nullptr;
    // cudaMalloc(&device_a, total * sizeof(cutlass::half_t));
    // cudaMalloc(&device_b, total * sizeof(cutlass::half_t));
    // cudaMalloc(&device_out, total * sizeof(cutlass::half_t));
    // cudaMemcpy(device_a, host_a.data(), total * sizeof(cutlass::half_t), cudaMemcpyHostToDevice);
    // cudaMemcpy(device_b, host_b.data(), total * sizeof(cutlass::half_t), cudaMemcpyHostToDevice);

    // const int threads = 256;
    // const int blocks = (total + threads - 1) / threads;
    // elementwise_add<<<blocks, threads>>>(device_a, device_b, device_out, total);
    // run_dummy_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass::half_t>>(device_out);
    
    run_block_attn_sam_smoke();
    // cudaError_t sync_err = cudaDeviceSynchronize();
    // if (sync_err != cudaSuccess) {
        // printf("Kernel sync error: %s\n", cudaGetErrorString(sync_err));
    // }
    // cudaFree(device_a);
    // cudaFree(device_b);
    // cudaFree(device_out);
    return 0;
}
