/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Adapted by Junxian Guo.
 * SAM-only kernel instantiation for head_dim=64, fp16.
 * This file is optimized for SAM-B which uses fp16 and head_dim=64.
 ******************************************************************************/

#include "namespace_config.h"
#include "flash_fwd_launch_sam.h"

namespace FLASH_NAMESPACE {

// Template instantiation for fp16, head_dim=64
// This is the primary configuration for SAM-B
template<>
void run_mha_fwd_sam_<cutlass::half_t, 64>(Flash_fwd_sam_params &params, cudaStream_t stream) {
    run_mha_fwd_sam_hdim64<cutlass::half_t>(params, stream);
}

}  // namespace FLASH_NAMESPACE
