/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Adapted by Junxian Guo.
 * Simplified launcher for vision transformer inference - no dropout, no causal.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"
#include <c10/cuda/CUDAException.h>

#include "static_switch.h"
#include "hardware_info.h"
#include "flash_params.h"
#include "flash_fwd_kernel_inference.h"

namespace FLASH_NAMESPACE {

////////////////////////////////////////////////////////////////////////////////////////////////////
// Simplified kernel launcher for vision transformers
// Template parameters reduced from 9 to 3:
// - Kernel_traits: Hardware configuration
// - Is_even_MN: Whether sequence lengths are multiples of block size
// - Is_even_K: Whether head dimension is multiple of alignment
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_even_MN, bool Is_even_K>
__global__ void flash_fwd_kernel(Flash_fwd_params params) {
    FLASH_NAMESPACE::compute_attn<Kernel_traits, Is_even_MN, Is_even_K>(params);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Launch wrapper - simplified version of run_flash_fwd_block
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;

    // Grid configuration: (num_m_blocks, batch, heads)
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);

    // Determine if sequences are evenly divisible by block sizes
    const bool is_even_MN = params.cu_seqlens_q == nullptr
                         && params.cu_seqlens_k == nullptr
                         && params.seqlen_k % Kernel_traits::kBlockN == 0
                         && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;

    // Launch kernel with runtime dispatch on evenness flags
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        BOOL_SWITCH(is_even_K, IsEvenKConst, [&] {
            // For SAM: always use optimized path when possible
            // IsEvenMNConst && IsEvenKConst = true when sequences are well-aligned
            auto kernel = &flash_fwd_kernel<
                Kernel_traits,
                IsEvenMNConst && IsEvenKConst,
                IsEvenKConst
            >;

            // Set shared memory size if needed
            if (smem_size >= 48 * 1024) {
                C10_CUDA_CHECK(cudaFuncSetAttribute(
                    kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    smem_size
                ));
            }

            // Launch kernel
            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Head dimension specific launchers
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void run_mha_fwd_sam_hdim32(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 32;
    // SAM doesn't use head_dim=32, but included for completeness
    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>>(params, stream);
}

template<typename T>
void run_mha_fwd_sam_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    // SAM-B uses head_dim=64
    // Block config: 128x128 (optimized for A100/H100)
    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>>(params, stream);
}

template<typename T>
void run_mha_fwd_sam_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;

    // Detect GPU architecture
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x = cc_major == 8 && cc_minor > 0;  // sm86, sm89

    if (is_sm8x) {
        // For sm86/89: 128x32 uses 48KB smem, allows 2 CTAs per SM
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>>(params, stream);
    } else {
        // For sm80 (A100), sm90 (H100): 128x64
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>>(params, stream);
    }
}

template<typename T>
void run_mha_fwd_sam_hdim256(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    // Larger head dimension: use smaller N blocks
    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>>(params, stream);
}

}  // namespace FLASH_NAMESPACE
