/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Adapted by Junxian Guo.
 * Vision transformer kernel instantiation for head_dim=256, bf16.
 ******************************************************************************/

#include "namespace_config.h"
#include "flash_fwd_launch.h"

namespace FLASH_NAMESPACE {

// Template instantiation for bf16, head_dim=256
template<>
void run_mha_fwd_<cutlass::bfloat16_t, 256>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim256<cutlass::bfloat16_t>(params, stream);
}

}  // namespace FLASH_NAMESPACE
