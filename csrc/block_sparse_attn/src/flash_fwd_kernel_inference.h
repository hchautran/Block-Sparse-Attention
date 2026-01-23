/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/
/******************************************************************************
 * Adapted by Hoai-Chau Tran   from https://github.com/Dao-AILab/flash-attention
 ******************************************************************************/
/******************************************************************************
 * Vision Transformer Simplified Version
 *
 * This file contains a simplified version of the flash attention forward kernel
 * optimized specifically for vision transformers usage.
 *
 * REMOVED FEATURES (not used by vision transformers inference):
 * - Dropout (Vision transformers inference never uses dropout)
 * - Causal masking (is_causal always false for vision transformers)
 * - Local window attention (not used in vision transformers)
 * - ALiBi positional bias (not used in vision models)
 * - Return softmax values (training-only feature)
 * - RNG/Philox seed handling (dropout-related, not needed)
 *
 * KEPT FEATURES (essential for vision transformers):
 * - Softmax computation and rescaling
 * - Support for varying sequence lengths (Is_even_MN, Is_even_K)
 *
 * USAGE:
 * - INFERENCE ONLY - no training support
 * - Always use with head_mask_type > 0 (block-sparse mode)
 * - Set is_causal=false, streaming_info=None
 * - Provide explicit base_blockmask for image-to-prompt patterns
 * - No dropout, no softmax return, no backward pass
 ******************************************************************************/

#pragma once

#include "namespace_config.h"

#include <cstdio>
#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"
#include "softmax.h"
#include "flash_blockmask.h"

namespace FLASH_NAMESPACE {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Simplified softmax for vision transformers block-sparse attention
// Always uses Check_inf=true since block-sparse may have skipped blocks
////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool Is_first, typename Tensor0, typename Tensor1, typename Tensor2>
inline __device__ void softmax_rescale_o_sam(
    Tensor0 &scores, Tensor1 &scores_max, Tensor1 &scores_sum,
    Tensor2 &acc_o, float softmax_scale_log2, bool Is_blocksparse_skip
) {
    if (Is_first) {
        FLASH_NAMESPACE::template reduce_max</*zero_init=*/true>(scores, scores_max);
        FLASH_NAMESPACE::scale_apply_exp2(scores, scores_max, softmax_scale_log2);
        FLASH_NAMESPACE::reduce_sum(scores, scores_sum);
    } else {
        Tensor scores_max_prev = make_fragment_like(scores_max);
        cute::copy(scores_max, scores_max_prev);
        FLASH_NAMESPACE::template reduce_max</*zero_init=*/false>(scores, scores_max);

        // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));

        #pragma unroll
        for (int mi = 0; mi < size(scores_max); ++mi) {
            // Handle -INFINITY from skipped blocks or empty blocks
            float scores_max_cur = Is_blocksparse_skip
                ? (scores_max(mi) == -INFINITY ? 0.0f : scores_max(mi))
                : scores_max(mi);
            float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
            scores_sum(mi) *= scores_scale;

            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
                acc_o_rowcol(mi, ni) *= scores_scale;
            }
        }

        FLASH_NAMESPACE::scale_apply_exp2(scores, scores_max, softmax_scale_log2);
        Tensor scores_sum_cur = make_fragment_like(scores_sum);
        FLASH_NAMESPACE::reduce_sum(scores, scores_sum_cur);

        #pragma unroll
        for (int mi = 0; mi < size(scores_sum); ++mi) {
            scores_sum(mi) += scores_sum_cur(mi);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Optional positional bias add (elementwise) before softmax
////////////////////////////////////////////////////////////////////////////////////////////////////


template<typename ElementAccum, typename TensorScores>
inline __device__ void scale_scores(TensorScores &scores, float scale) {
    #pragma unroll
    for (int mi = 0; mi < size<0>(scores); ++mi) {
        #pragma unroll
        for (int ni = 0; ni < size<1>(scores); ++ni) {
            #pragma unroll
            for (int ki = 0; ki < size<2>(scores); ++ki) {
                scores(mi, ni, ki) = static_cast<ElementAccum>(scores(mi, ni, ki) * scale);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Main vision transformers block-sparse attention kernel
//
// Template parameters:
//   - Kernel_traits: Hardware-specific kernel configuration
//   - Is_even_MN: Whether sequence lengths are multiples of block size
//   - Is_even_K: Whether head dimension is multiple of block size
//   - Params: Parameter struct containing all kernel arguments
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_even_MN, bool Is_even_K, typename Params>
inline __device__ auto load_pos(
    const Params &params, const int bidb, const int bidh, const int m_block, const int n_block
) {
    using Element = typename Kernel_traits::Element;
    using index_t = typename Kernel_traits::index_t;
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    const int tidx = threadIdx.x;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    const index_t row_offset_pos = params.pos_batch_stride  * bidb
        + bidh * params.pos_head_stride
        + m_block * kBlockM * params.pos_row_stride
        + n_block * kBlockN * params.pos_col_stride;

    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.pos_ptr) + row_offset_pos),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.pos_row_stride, params.pos_col_stride));

 
    return gP;

}

template<typename ElementAccum, typename TensorScores, typename TensorPos>
inline __device__ void add_pos_bias(TensorScores &scores, TensorPos const &pos) {
    #pragma unroll
    for (int mi = 0; mi < size<0>(scores); ++mi) {
        #pragma unroll
        for (int ni = 0; ni < size<1>(scores); ++ni) {
            scores(mi, ni) += static_cast<ElementAccum>(pos(mi, ni));
        }
    }
}

template<typename Kernel_traits, bool Is_even_MN, typename Params,
         typename BlockInfoT, typename TensorScores, typename GmemTiledCopyP, typename GmemThrCopyP>
inline __device__ void store_attn_block(
    const Params &params, const BlockInfoT &binfo,
    const GmemTiledCopyP &gmem_tiled_copy_P, const GmemThrCopyP &gmem_thr_copy_P,
    const TensorScores &scores, const int bidb, const int bidh, const int m_block, const int n_block
) {
    if (params.attn_ptr == nullptr) return;

    using Element = typename Kernel_traits::Element;
    using index_t = typename Kernel_traits::index_t;
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;

    const index_t row_offset_attn = binfo.q_offset(params.attn_batch_stride, params.attn_row_stride, bidb)
        + bidh * params.attn_head_stride
        + m_block * kBlockM * params.attn_row_stride
        + n_block * kBlockN * params.attn_col_stride;

    Tensor gAttn = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.attn_ptr) + row_offset_attn),
                               Shape<Int<kBlockM>, Int<kBlockN>>{},
                               make_stride(params.attn_row_stride, params.attn_col_stride));
    Tensor tPgAttn = gmem_thr_copy_P.partition_D(gAttn);
    Tensor rAttn = FLASH_NAMESPACE::convert_type<Element>(scores);
    Tensor tPrAttn = gmem_thr_copy_P.partition_S(rAttn);

    Tensor cAttn = make_identity_tensor(make_shape(size<0>(gAttn), size<1>(gAttn)));
    Tensor tPcAttn = gmem_thr_copy_P.partition_D(cAttn);
    Tensor tPpAttn = make_tensor<bool>(make_shape(size<2>(tPgAttn)));

    if (!Is_even_MN) {
        const int max_col = binfo.actual_seqlen_k - n_block * kBlockN;
        #pragma unroll
        for (int k = 0; k < size(tPpAttn); ++k) {
            tPpAttn(k) = get<1>(tPcAttn(0, 0, k)) < max_col;
        }
    }

    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_MN, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_P, tPrAttn, tPgAttn, tPcAttn, tPpAttn, binfo.actual_seqlen_q - m_block * kBlockM
    );
}

template<typename Kernel_traits, bool Is_even_MN, bool Is_even_K, typename Params>
inline __device__ void compute_block_attn_sam(
    const Params &params, const int bidb, const int bidh, const int m_block
) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory
    extern __shared__ char smem_[];

    // Thread index
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int MMA_M = kBlockM / decltype(size<0>(typename Kernel_traits::TiledMma::TiledShape_MNK{}))::value;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    // SAM is non-causal and non-local, so n_block_min is always 0
    const int n_block_min = 0;
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);

    // Initialize block-sparse mask iterator (always use fwdBlockmask for vision transformers)
    fwdBlockmask blockmask(params, binfo, kBlockM, kBlockN, bidb, bidh, m_block, n_block_min, n_block_max);

    int max_block_idx = blockmask.max_block_idx;
    int max_no_larger_idx = blockmask.max_no_larger(n_block_max - 1);

    // Check if this row has any valid blocks to attend to
    bool empty_line_flag = n_block_max <= n_block_min;
    empty_line_flag = empty_line_flag || max_no_larger_idx == -1 || blockmask.mask_val(max_no_larger_idx) < n_block_min;

    __syncthreads();

    // Early exit if no valid attention blocks for this query row
    if (empty_line_flag) {
        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
            + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;

        Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                                Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                make_stride(params.o_row_stride, _1{}));  // [kBlockM, kHeadDim]
        Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                                  Shape<Int<kBlockM>>{}, Stride<_1>{});   // [kBlockM]

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);           // per-thread view of gO
        Tensor tOrO = make_tensor<Element>(shape(tOgO));         // zeroed register tile for O
        clear(tOrO);

        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO))); // [kBlockM, kHeadDim] identity coords
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                          // per-thread coords for O
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));             // K predication mask

        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }

        FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );

        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) {
                gLSE(row) = INFINITY;
            }
        }
        return;
    }

    // Setup global memory tensor pointers
    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));


    // Setup shared memory tensors
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                            typename Kernel_traits::SmemLayoutKV{});  // [kBlockN, kHeadDim] in smem
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    // Setup copy operations
    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopyP gmem_tiled_copy_P;
    auto gmem_thr_copy_P = gmem_tiled_copy_P.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    // Setup MMA
    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);

    ElementAccum c_dummy_o;
    auto c_tensor_o = make_tensor(make_rmem_ptr<ElementAccum>(&c_dummy_o),
                                  make_layout(Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    Tensor acc_o = thr_mma.partition_fragment_C(c_tensor_o);

    ElementAccum c_dummy_s;
    auto c_tensor_s = make_tensor(make_rmem_ptr<ElementAccum>(&c_dummy_s),
                                  make_layout(Shape<Int<kBlockM>, Int<kBlockN>>{}));

    const bool use_pos = params.pos_ptr != nullptr;
    const bool return_attn = params.attn_ptr != nullptr;

    auto gP = load_pos<Kernel_traits, Is_even_MN, Is_even_K>( 
        params, bidb, bidh, m_block, (n_block_max - 1)
    );
    Tensor tSgP = thr_mma.partition_C(gP);
                                  
    // if (thread0()) {

    //     // gP.data() = gP.data() + (-index_t(kBlockN * 2 * params.pos_col_stride));
    //     auto cta_tiler = make_shape(Int<128>{}, Int<128>{});   
    //     auto cta_coord = make_coord(m_block, 1);     
    //     Tensor g_P = local_tile(gP, cta_tiler, cta_coord, Step<_1,_1>{}); 
    //     Tensor tSg_P = thr_mma.partition_C(g_P);
    //     print(g_P);
    //     tSg_P.data() = tSgP.data() + (-index_t(kBlockN * 31 * params.pos_col_stride));
    //     Tensor acc_s = thr_mma.partition_fragment_C(c_tensor_s); 
    //     axpby(1.0, tSg_P, 1.0 ,acc_s);


    //     printf("\n========\n");
    //     print(gP);
    //     printf("\n========\n");
    //     print(tSg_P);
  
        
    //     printf("\n========\n");
    //     for (int k = 0; k < size<2>(acc_s); ++k) {
    //         for (int j = 0; j < size<1>(acc_s); ++j) {
    //             for (int i = 0; i < size<0>(acc_s); ++i) {
    //                 printf("%f ", static_cast<float>(acc_s(i,j, k)));

    //             }
    //         }
    //     }
    //     printf("\n========\n");
    // }

    // Copy operations setup
    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ); // smem Q tile for cp.async

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt); // smem V^T tile for MMA

    // Allocate softmax statistics
    Tensor scores_max = make_tensor<ElementAccum>(Shape<Int<2 * size<1>(acc_o)>>{});
    Tensor scores_sum = make_fragment_like(scores_max);

    // Identity tensors for predication
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));

    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);   // per-thread Q coords
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV); // per-thread K/V coords

    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // === PROLOGUE: Load Q ===
    Tensor tQrQ = make_fragment_like(tQgQ);
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                                   binfo.actual_seqlen_q - m_block * kBlockM);
    if (Kernel_traits::Is_Q_in_regs) { cute::cp_async_fence(); }

    if (Kernel_traits::Share_Q_K_smem) {
        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ); // Q fragment view for MMA
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    // Load first K block
    int n_block = n_block_max - 1;
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV,
                                                   binfo.actual_seqlen_k - n_block * kBlockN);
    cute::cp_async_fence();

    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        FLASH_NAMESPACE::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ); // Q fragment view for MMA
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);

    // === BLOCK-SPARSE ATTENTION LOOP ===
    // Uses non-causal, so n_masking_steps = 1
    constexpr int n_masking_steps = 1;

    // Initialize block-sparse iteration state
    int mask_block_idx = max_no_larger_idx;
    int mask_val = mask_block_idx == -1 ? -1 : blockmask.mask_val(mask_block_idx);
    bool is_last_block = mask_val == -1;
    int next_block_col_idx = mask_val;
    int leap = 0;

    // === First masking iteration (handles last K block which may have OOB elements) ===
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        bool is_skip = n_block != next_block_col_idx;

        if (is_skip) {
            // This K block is masked out (not in sparse pattern)
            leap = (masking_step + 1 == n_masking_steps) ? n_block - next_block_col_idx : 1;
            leap = is_last_block ? 0 : leap;

            Tensor acc_s = thr_mma.partition_fragment_C(c_tensor_s); // [kBlockM, kBlockN]

            
            clear(acc_s);
            FLASH_NAMESPACE::cp_async_wait<0>();
            __syncthreads();

            // Advance V
            if (masking_step > 0) {
                tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));

                FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
            } else {
                FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                    gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
                );
            }
            cute::cp_async_fence();

            // Scores are all masked, apply mask with 0 valid elements
            scale_scores<ElementAccum>(acc_s, params.scale_softmax);


            FLASH_NAMESPACE::cp_async_wait<0>();
            __syncthreads();

            if (use_pos) {
                axpby(1.0, tSgP,1.0 ,acc_s);
            }
            Tensor scores = make_tensor(acc_s.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_s.layout())); // [kBlockM, kBlockN]

            if (n_block > n_block_min && !is_last_block) {
                tKgK.data() = tKgK.data() + (-index_t(kBlockN * leap * params.k_row_stride));
                tSgP.data() = tSgP.data() + (-index_t(kBlockN * leap * params.pos_col_stride));
                FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);
                cute::cp_async_fence();
            }

            // Update softmax statistics with empty block
            const float softmax_scale_log2 = M_LOG2E;
            masking_step == 0
                ? softmax_rescale_o_sam</*Is_first=*/true>(scores, scores_max, scores_sum, acc_o, softmax_scale_log2, /*Is_blocksparse_skip=*/true)
                : softmax_rescale_o_sam</*Is_first=*/false>(scores, scores_max, scores_sum, acc_o, softmax_scale_log2, /*Is_blocksparse_skip=*/true);

            if (n_masking_steps > 1 && n_block <= n_block_min) {
                --n_block;
                break;
            }
        } else {
            // This K block is NOT skipped (part of sparse pattern)
            if (!is_last_block) {
                mask_block_idx++;
                mask_val = blockmask.mask_val(mask_block_idx);
                is_last_block = mask_block_idx >= max_block_idx || mask_val == -1;
                next_block_col_idx = is_last_block ? -1 : mask_val;
            }

            leap = (masking_step + 1 == n_masking_steps) ? n_block - next_block_col_idx : 1;
            leap = is_last_block ? 0 : leap;

            Tensor acc_s = thr_mma.partition_fragment_C(c_tensor_s); // [kBlockM, kBlockN]
            clear(acc_s);
            FLASH_NAMESPACE::cp_async_wait<0>();
            __syncthreads();

            // Advance V
            if (masking_step > 0) {
                tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
                FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
            } else {
                FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                    gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
                );
            }
            cute::cp_async_fence();

            // Compute S = Q @ K^T
            FLASH_NAMESPACE::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
                acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K
            );

                    // Reshape and compute softmax
            scale_scores<ElementAccum>(acc_s, params.scale_softmax);
            if (use_pos) {
                axpby(1.0, tSgP, 1.0 ,acc_s);
            }

            Tensor scores = make_tensor(acc_s.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_s.layout()));


            // Apply masking for OOB elements (Vision transformers is non-causal, non-local)
            if (!Is_even_MN) {
                FLASH_NAMESPACE::apply_mask(scores, binfo.actual_seqlen_k - n_block * kBlockN);
            }

            FLASH_NAMESPACE::cp_async_wait<0>();
            __syncthreads();

            if (n_block > n_block_min && !is_last_block) {
                tKgK.data() = tKgK.data() + (-index_t(kBlockN * leap * params.k_row_stride));
                tSgP.data() = tSgP.data() + (-index_t(kBlockN * leap * params.pos_col_stride));
                FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);
                cute::cp_async_fence();
            }

            // Compute softmax and rescale previous O
            const float softmax_scale_log2 = M_LOG2E;
            masking_step == 0
                ? softmax_rescale_o_sam</*Is_first=*/true>(scores, scores_max, scores_sum, acc_o, softmax_scale_log2, /*Is_blocksparse_skip=*/false)
                : softmax_rescale_o_sam</*Is_first=*/false>(scores, scores_max, scores_sum, acc_o, softmax_scale_log2, /*Is_blocksparse_skip=*/false);

            // Convert scores to element type (fp16/bf16)
            Tensor rP = FLASH_NAMESPACE::convert_type<Element>(scores); // [kBlockM, kBlockN] in fp16/bf16
            Tensor tOrP = make_tensor(rP.data(), FLASH_NAMESPACE::convert_layout_rowcol_Aregs<Kernel_traits::TiledMma>(rP.layout())); // MMA A regs

            // Compute O = P @ V
            FLASH_NAMESPACE::gemm_A_in_regs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

            if (n_masking_steps > 1 && n_block <= n_block_min) {
                --n_block;
                break;
            }
        }
    }

    // === Remaining blocks (no masking needed for OOB) ===
    leap = n_block - next_block_col_idx;
    if (!is_last_block) {
        tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
        n_block = next_block_col_idx;
    }

    // Iterate over remaining blocks in sparse pattern
    // todo: explain how KV tile is loaded here 
    for (; !is_last_block && n_block >= n_block_min; n_block = next_block_col_idx) {
        ++mask_block_idx;
        mask_val = blockmask.mask_val(mask_block_idx);
        is_last_block = mask_block_idx >= max_block_idx || mask_val == -1;
        next_block_col_idx = mask_val;

        Tensor acc_s = thr_mma.partition_fragment_C(c_tensor_s); // [kBlockM, kBlockN]
        clear(acc_s);
        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();

        // Advance V with leap
        tVgV.data() = tVgV.data() + (-index_t(kBlockN * leap * params.v_row_stride));
        FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
        cute::cp_async_fence();

        // Compute S = Q @ K^T
        FLASH_NAMESPACE::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();

        leap = n_block - next_block_col_idx;

        scale_scores<ElementAccum>(acc_s, params.scale_softmax);
        if (use_pos) {
            axpby(1.0, tSgP,1.0 ,acc_s);
        }

        if (!is_last_block) {
            tKgK.data() = tKgK.data() + (-index_t(kBlockN * leap * params.k_row_stride));
            tSgP.data() = tSgP.data() + (-index_t(kBlockN * leap * params.pos_col_stride));
            FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);
            cute::cp_async_fence();
        }


        Tensor scores = make_tensor(acc_s.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_s.layout())); // [kBlockM, kBlockN]
        softmax_rescale_o_sam</*Is_first=*/false>(scores, scores_max, scores_sum, acc_o, M_LOG2E, /*Is_blocksparse_skip=*/false);

        // Convert scores to element type and compute O = P @ V
        Tensor rP = FLASH_NAMESPACE::convert_type<Element>(scores);
        Tensor tOrP = make_tensor(rP.data(), FLASH_NAMESPACE::convert_layout_rowcol_Aregs<Kernel_traits::TiledMma>(rP.layout()));

        FLASH_NAMESPACE::gemm_A_in_regs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }



    // === EPILOGUE: Write O and LSE to global memory ===

    Tensor acc_o_rowcol = make_tensor(acc_o.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout())); // [kBlockM, kHeadDim]
    Tensor lse = make_fragment_like(scores_sum); // per-row LSE

    #pragma unroll
    for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
        float sum = scores_sum(mi);
        float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
        lse(mi) = (sum == 0.f || sum != sum) ? INFINITY : scores_max(mi) + __logf(sum);

        #pragma unroll
        for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
            acc_o_rowcol(mi, ni) *= inv_sum;
        }
    }

    // Convert O to fp16/bf16
    Tensor rO = FLASH_NAMESPACE::convert_type<Element>(acc_o); // [kBlockM, kHeadDim] in fp16/bf16
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{}); // [kBlockM, kHeadDim] in smem

    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

    if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;

    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.o_row_stride, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                              Shape<Int<kBlockM>>{}, Stride<_1>{});   // [kBlockM]

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO)); // per-thread O regs for gmem store
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    // Write LSE
    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{}); // [kBlockM, kHeadDim] coords
    Tensor taccOcO = thr_mma.partition_C(caccO);
    static_assert(decltype(size<0>(taccOcO))::value == 4);
    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));

    if (get<1>(taccOcO_row(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
        }
    }

    // Write O
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));

    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    }

    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Entry point kernel for vision transformers block-sparse attention (inference only)
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_even_MN, bool Is_even_K, typename Params>
inline __device__ void compute_attn(const Params &params) {
    const int m_block = blockIdx.x;
    const int bidb = blockIdx.y;
    const int bidh = blockIdx.z;

    // Uses block-sparse attention (head_mask_type > 0)
    // No need to check head_mask_type - always call block-sparse kernel
    FLASH_NAMESPACE::compute_block_attn_sam<Kernel_traits, Is_even_MN, Is_even_K>(
        params, bidb, bidh, m_block
    );
}

}  // namespace FLASH_NAMESPACE
