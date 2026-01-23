/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Adapted by Hoai-Chau Tran
 * Optimized for vision transformer inference.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"

#include <cuda.h>
#include <vector>

namespace FLASH_NAMESPACE {

////////////////////////////////////////////////////////////////////////////////////////////////////
// Simplified parameter struct for vision transformer inference
// Removed: dropout, causal, window attention, ALiBi, rotary, backward pass
////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_params {
    using index_t = int64_t;

    // ============================================================================
    // Input/Output Pointers
    // ============================================================================
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;
    void *__restrict__ o_ptr;

    // Softmax statistics (LSE = Log-Sum-Exp)
    void *__restrict__ softmax_lse_ptr;

    // ============================================================================
    // Strides (all in elements, not bytes)
    // ============================================================================
    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t o_batch_stride;

    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t o_row_stride;

    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;
    index_t o_head_stride;

    // ============================================================================
    // Dimensions
    // ============================================================================
    int b;              // batch size
    int h;              // number of query heads
    int h_k;            // number of key/value heads (for MQA/GQA)
    int h_h_k_ratio;    // h / h_k (precomputed)
    int d;              // head dimension
    int d_rounded;      // head dimension rounded to 32 or 64

    int seqlen_q;           // query sequence length
    int seqlen_k;           // key/value sequence length
    int seqlen_q_rounded;   // rounded to block size (128)
    int seqlen_k_rounded;   // rounded to block size (128)

    int total_q;        // total query tokens across batch (for varlen)

    // ============================================================================
    // Varlen support (variable-length sequences)
    // ============================================================================
    // Array of length b+1 holding starting offset of each sequence
    int *__restrict__ cu_seqlens_q;
    int *__restrict__ cu_seqlens_k;

    // KV-cache related (not used for inference-only, but needed by BlockInfo)
    // Set these to nullptr/0 for inference
    int *__restrict__ seqused_k;     // Actual used length of each K sequence (nullptr)
    void *__restrict__ knew_ptr;     // New K values for cache append (nullptr)
    int seqlen_knew;                 // Length of new K (0)

    // ============================================================================
    // Block-sparse attention specific
    // ============================================================================
    int *__restrict__ blockmask;        // Block-sparse mask
    int *__restrict__ head_mask_type;   // Per-head mask type (0=dense, 1=sparse)

    int m_block_dim;                // M block dimension (default 128)
    int n_block_dim;                // N block dimension (default 128)
    int num_blocksparse_heads;      // Number of heads using sparse pattern

    // ============================================================================
    // Optional positional matrix (added to attention scores before softmax)
    // Layout expected: [b, h, seqlen_q, seqlen_k]
    // ============================================================================
    void *__restrict__ pos_ptr;
    index_t pos_batch_stride;
    index_t pos_head_stride;
    index_t pos_row_stride;
    index_t pos_col_stride;

    // ============================================================================
    // Optional attention map output
    // Layout expected: [b, h, seqlen_q, seqlen_k]
    // Stored in Element type (fp16/bf16) for bandwidth efficiency.
    // ============================================================================
    void *__restrict__ attn_ptr;
    index_t attn_batch_stride;
    index_t attn_head_stride;
    index_t attn_row_stride;
    index_t attn_col_stride;

    // ============================================================================
    // Scaling factors
    // ============================================================================
    float scale_softmax;        // 1 / sqrt(d)
    float scale_softmax_log2;   // scale_softmax * log2(e)

    // ============================================================================
    // Flags
    // ============================================================================
    bool is_bf16;   // true if bfloat16, false if float16

    // For varlen: if true, seqlen_k = cu_seqlens_k[bidb+1] - cu_seqlens_k[bidb]
    // Otherwise, seqlen_k = cu_seqlens_k[bidb]
    bool is_seqlens_k_cumulative;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Template declarations for forward kernels
////////////////////////////////////////////////////////////////////////////////////////////////////

// Only non-causal version needed for inference
template<typename T, int Headdim>
void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream);

}  // namespace FLASH_NAMESPACE
