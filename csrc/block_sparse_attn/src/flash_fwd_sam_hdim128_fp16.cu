/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Adapted by Junxian Guo.
 * SAM-only kernel instantiation for head_dim=128, fp16.
 * Useful for SAM-L/H variants.
 ******************************************************************************/

#include "namespace_config.h"
#include "flash_fwd_launch_sam.h"

namespace FLASH_NAMESPACE {

// Template instantiation for fp16, head_dim=128
template<>
void run_mha_fwd_sam_<cutlass::half_t, 128>(Flash_fwd_sam_params &params, cudaStream_t stream) {
    run_mha_fwd_sam_hdim128<cutlass::half_t>(params, stream);
}

}  // namespace FLASH_NAMESPACE
