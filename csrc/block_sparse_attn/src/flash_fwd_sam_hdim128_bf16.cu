/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Adapted by Junxian Guo.
 * SAM-only kernel instantiation for head_dim=128, bf16.
 ******************************************************************************/

#include "namespace_config.h"
#include "flash_fwd_launch_sam.h"

namespace FLASH_NAMESPACE {

// Template instantiation for bf16, head_dim=128
template<>
void run_mha_fwd_sam_<cutlass::bfloat16_t, 128>(Flash_fwd_sam_params &params, cudaStream_t stream) {
    run_mha_fwd_sam_hdim128<cutlass::bfloat16_t>(params, stream);
}

}  // namespace FLASH_NAMESPACE
