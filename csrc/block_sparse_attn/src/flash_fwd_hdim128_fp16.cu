/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Adapted by Junxian Guo.
 * Vision transformer kernel instantiation for head_dim=128, fp16.
 * Useful for vision transformers-L/H variants.
 ******************************************************************************/

#include "namespace_config.h"
#include "flash_fwd_launch.h"

namespace FLASH_NAMESPACE {

// Template instantiation for fp16, head_dim=128
template<>
void run_mha_fwd_<cutlass::half_t, 128>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim128<cutlass::half_t>(params, stream);
}

}  // namespace FLASH_NAMESPACE
