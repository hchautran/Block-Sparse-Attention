/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Adapted by Junxian Guo.
 * SAM-only kernel instantiation for head_dim=32, fp16.
 ******************************************************************************/

#include "namespace_config.h"
#include "flash_fwd_launch_sam.h"

namespace FLASH_NAMESPACE {

// Template instantiation for fp16, head_dim=32
template<>
void run_mha_fwd_sam_<cutlass::half_t, 32>(Flash_fwd_sam_params &params, cudaStream_t stream) {
    run_mha_fwd_sam_hdim32<cutlass::half_t>(params, stream);
}

}  // namespace FLASH_NAMESPACE
