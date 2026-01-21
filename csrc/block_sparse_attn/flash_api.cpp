/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Adapted by Junxian Guo.
 * Block Sparse Attention API - optimized for vision transformer inference.
 *
 * Removed: backward pass, dropout, causal masking, streaming, window attention
 ******************************************************************************/

#include <torch/python.h>
#include <torch/nn/functional.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cutlass/numeric_types.h>

#include "namespace_config.h"
#include "src/hardware_info.h"
#include "src/flash_params.h"
#include "src/static_switch.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace FLASH_NAMESPACE {

////////////////////////////////////////////////////////////////////////////////////////////////////
// Set parameters for SAM forward pass
////////////////////////////////////////////////////////////////////////////////////////////////////

void set_params_fprop(
    Flash_fwd_params &params,
    // Sizes
    const size_t b,
    const size_t seqlen_q,
    const size_t seqlen_k,
    const size_t seqlen_q_rounded,
    const size_t seqlen_k_rounded,
    const size_t h,
    const size_t h_k,
    const size_t d,
    const size_t d_rounded,
    // Device pointers
    const at::Tensor q,
    const at::Tensor k,
    const at::Tensor v,
    at::Tensor out,
    void *cu_seqlens_q_d,
    void *cu_seqlens_k_d,
    void *softmax_lse_d,
    float softmax_scale
) {
    // Reset parameters
    params = {};

    params.is_bf16 = q.dtype() == torch::kBFloat16;

    // Set pointers and strides (all in elements, not bytes)
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.o_ptr = out.data_ptr();

    params.q_row_stride = q.stride(-3);
    params.k_row_stride = k.stride(-3);
    params.v_row_stride = v.stride(-3);
    params.o_row_stride = out.stride(-3);

    params.q_head_stride = q.stride(-2);
    params.k_head_stride = k.stride(-2);
    params.v_head_stride = v.stride(-2);
    params.o_head_stride = out.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = q.stride(0);
        params.k_batch_stride = k.stride(0);
        params.v_batch_stride = v.stride(0);
        params.o_batch_stride = out.stride(0);
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);

    // KV-cache not used for SAM (set to nullptr/0)
    params.seqused_k = nullptr;
    params.knew_ptr = nullptr;
    params.seqlen_knew = 0;

    params.softmax_lse_ptr = softmax_lse_d;

    // Set dimensions
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set scaling factors
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;

    params.is_seqlens_k_cumulative = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Dispatcher - simplified version with only dtype and headdim switches
////////////////////////////////////////////////////////////////////////////////////////////////////

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            run_mha_fwd_<elem_type, kHeadDim>(params, stream);
        });
    });
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Main SAM attention forward function
////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr int SPARSE_SIZE = 128;

std::vector<at::Tensor>
mha_varlen_fwd(
    at::Tensor &q,                       // total_q x num_heads x head_size
    const at::Tensor &k,                 // total_k x num_heads_k x head_size
    const at::Tensor &v,                 // total_k x num_heads_k x head_size
    const at::Tensor &cu_seqlens_q,      // b+1
    const at::Tensor &cu_seqlens_k,      // b+1
    const at::Tensor &head_mask_type,    // num_heads
    std::optional<at::Tensor> &row_blockmask_,  // batch, num_sparse_heads, nrow, ncol
    std::optional<at::Tensor> &positional_,     // batch, num_heads, seqlen_q, seqlen_k
    int max_seqlen_q,
    const int max_seqlen_k,
    const float softmax_scale
) {
    // Set device guard
    at::cuda::CUDAGuard device_guard{q.device()};

    // Check GPU compatibility
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x_min = cc_major >= 8;
    TORCH_CHECK(is_sm8x_min, "SAM Block Sparse Attention requires Ampere GPUs or newer (sm80+).");

    // Handle optional blockmask
    const bool has_blockmask = row_blockmask_.has_value();
    at::Tensor row_blockmask;
    if (has_blockmask) {
        row_blockmask = row_blockmask_.value();
    }
    const bool has_positional = positional_.has_value();
    at::Tensor positional;
    if (has_positional) {
        positional = positional_.value();
    }

    // Check dtypes
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "SAM attention only supports fp16 and bf16");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must be int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must be int32");
    TORCH_CHECK(head_mask_type.dtype() == torch::kInt32, "head_mask_type must be int32");

    if (has_blockmask) {
        TORCH_CHECK(row_blockmask.dtype() == torch::kInt32, "row_blockmask must be int32");
    }
    if (has_positional) {
        TORCH_CHECK(positional.dtype() == q_dtype, "positional must have the same dtype as q");
    }

    // Check device placement
    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(cu_seqlens_q);
    CHECK_DEVICE(cu_seqlens_k);
    CHECK_DEVICE(head_mask_type);
    if (has_blockmask) {
        CHECK_DEVICE(row_blockmask);
    }
    if (has_positional) {
        CHECK_DEVICE(positional);
    }

    // Check contiguity
    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);
    CHECK_CONTIGUOUS(head_mask_type);
    if (has_blockmask) {
        CHECK_CONTIGUOUS(row_blockmask);
    }
    if (has_positional) {
        TORCH_CHECK(positional.stride(-1) == 1, "positional must have contiguous last dimension");
    }

    // Extract dimensions
    const auto sizes = q.sizes();
    const int total_q = sizes[0];
    const int batch_size = cu_seqlens_q.numel() - 1;
    const int num_heads = sizes[1];
    const int head_size = sizes[2];
    const int num_heads_k = k.size(1);
    int num_blocksparse_heads = 0;

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size <= 256, "SAM attention supports head dimension at most 256");
    TORCH_CHECK(head_size % 8 == 0, "head dimension must be a multiple of 8");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    // Check shapes
    CHECK_SHAPE(q, total_q, num_heads, head_size);
    const int total_k = k.size(0);
    CHECK_SHAPE(k, total_k, num_heads_k, head_size);
    CHECK_SHAPE(v, total_k, num_heads_k, head_size);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
    CHECK_SHAPE(head_mask_type, num_heads);

    // Create output tensor
    at::Tensor out = torch::empty_like(q);

    // Round dimensions
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, head_size <= 128 ? 32 : 64);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, SPARSE_SIZE);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, SPARSE_SIZE);

    // Check blockmask shape if provided
    if (has_blockmask) {
        num_blocksparse_heads = row_blockmask.size(1);
        CHECK_SHAPE(row_blockmask, batch_size, num_blocksparse_heads,
                    seqlen_q_rounded / SPARSE_SIZE, seqlen_k_rounded / SPARSE_SIZE);
    }
    if (has_positional) {
        CHECK_SHAPE(positional, batch_size, num_heads, max_seqlen_q, max_seqlen_k);
    }

    // Create softmax LSE tensor
    auto opts = q.options();
    auto softmax_lse = torch::empty({batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));

    // Set up parameters
    Flash_fwd_params params;
    set_params_fprop(
        params,
        batch_size,
        max_seqlen_q, max_seqlen_k,
        seqlen_q_rounded, seqlen_k_rounded,
        num_heads, num_heads_k,
        head_size, head_size_rounded,
        q, k, v, out,
        cu_seqlens_q.data_ptr(),
        cu_seqlens_k.data_ptr(),
        softmax_lse.data_ptr(),
        softmax_scale
    );

    params.total_q = total_q;
    params.head_mask_type = static_cast<int *>(head_mask_type.data_ptr());

    if (has_blockmask) {
        params.blockmask = static_cast<int *>(row_blockmask.data_ptr());
        params.m_block_dim = SPARSE_SIZE;
        params.n_block_dim = SPARSE_SIZE;
        params.num_blocksparse_heads = num_blocksparse_heads;
    } else {
        params.blockmask = nullptr;
    }
    if (has_positional) {
        params.pos_ptr = positional.data_ptr();
        params.pos_batch_stride = positional.stride(0);
        params.pos_head_stride = positional.stride(1);
        params.pos_row_stride = positional.stride(2);
        params.pos_col_stride = positional.stride(3);
    } else {
        params.pos_ptr = nullptr;
        params.pos_batch_stride = 0;
        params.pos_head_stride = 0;
        params.pos_row_stride = 0;
        params.pos_col_stride = 0;
    }

    // Run kernel
    if (max_seqlen_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd(params, stream);
    } else {
        // Empty tensor case
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    return {out, softmax_lse};
}

}  // namespace FLASH_NAMESPACE

////////////////////////////////////////////////////////////////////////////////////////////////////
// PyBind11 module export
////////////////////////////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Block Sparse Attention (Inference Only)";
    m.def("fwd", &FLASH_NAMESPACE::mha_varlen_fwd, "Forward pass (inference only)");
}
