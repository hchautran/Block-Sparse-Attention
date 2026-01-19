import torch
import json
import os
from block_sparse_attn import block_sparse_attn_func
import torch.nn.functional as F
from utils import time_fwd, flops, efficiency, write_to_excel

BLOCK_SIZE = int(os.getenv("BLOCK_SPARSE_ATTN_BLOCK_SIZE", "128"))

def measure_peak_memory_mb(func, *args, **kwargs):
    device = args[0].device if args else torch.device("cuda")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    mem_before = torch.cuda.memory_allocated(device)
    with torch.no_grad():
        func(*args, **kwargs)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    return max(0, peak - mem_before) / (1024 ** 2)

def generate_base_sparsity_mask(max_seqlen_q, max_seqlen_k, round_base, m_block_dim, n_block_dim, sparsity, causal=False, device="cuda"):
    def round_to_multiple(x, base):
        return ((x + base - 1) // base) * base
        
    nrow = round_to_multiple(max_seqlen_q, round_base) // m_block_dim
    ncol = round_to_multiple(max_seqlen_k, round_base) // n_block_dim
    base_mask = torch.zeros(1, nrow, ncol, device=device, dtype=torch.bool)
    
    density = 1.0 - sparsity
    if 0.0 < density < 1.0:
        for i in range(nrow): 
            idx = nrow - i - 1
            if causal:
                available_col_num = max(0, ncol - i)
                num_one = max(1, int(density * available_col_num))
                if available_col_num > 0:
                    base_mask[0][idx, torch.randperm(available_col_num)[:num_one]] = True
            else:
                available_col_num = ncol
                num_one = max(1, int(density * available_col_num))
                base_mask[0][idx, torch.randperm(available_col_num)[:num_one]] = True
    elif density == 1.0:
        base_mask[0] = torch.ones_like(base_mask[0])
    
    calculated_block_num = base_mask.sum().item()
    total_block_num = 0
    if causal:
        for i in range(nrow):
             total_block_num += max(0, ncol - i)
    else:
        total_block_num = nrow * ncol
        
    real_sparsity = 1.0 - (calculated_block_num / total_block_num) if total_block_num > 0 else 0.0
    return base_mask, real_sparsity

def get_sparsity_list(sampling_steps, seqlen, causal):
    blockmask_element_num = (seqlen // BLOCK_SIZE) ** 2 // (2 if causal else 1)
    stride = max(blockmask_element_num // sampling_steps, 1)
    actual_steps = (blockmask_element_num + stride - 1) // stride
    sparsity_list = []
    for i in range(actual_steps):
        sparse_rate = (1 + i * stride) / blockmask_element_num
        if 0.0 <= sparse_rate <= 0.95:
            sparsity_list.append(sparse_rate)
    return sparsity_list

def profile_blocksparse_fwd():
    # Configuration
    repeats = 15
    block_sparse_repeats = 3
    device = 'cuda:0'
    dtype = torch.float16
    causal = False
    batch_size = 200 
    sparsity_sampling_steps = 10
    seqlen_vals = [1024]
    headdim = 64 
    dim = 1024 
    dropout_p = 0.0
    
    nheads = dim // headdim
    excel_dir_path = "./excel/blocksparse/"
    excel_file_name = f"hdim{headdim}_nheads{nheads}_bts{batch_size}_fwd" + ("_causal" if causal else "")
    excel_label = [
        "batch_size",
        "seqlen",
        "actual_sparsity",
        "speed",
        "latency",
        "peak_mem_mb",
        "speedup",
        "base_speed",
        "base_latency",
        "base_peak_mem_mb",
    ]
    excel_data = []
    all_results = {}

    print(f"\nConfiguration: headdim={headdim}, nheads={nheads}, batch_size={batch_size}, causal={causal}")
    
    for seqlen in seqlen_vals:
        print(f"\n{'='*40} SeqLen: {seqlen} {'='*40}")
        print(f"{'Sparsity':<10} | {'SDPA Latency':<15} | {'SDPA TFLOPs':<15} | {'SDPA Mem':<12} | {'Avg Latency':<15} | {'Avg TFLOPs':<15} | {'Block Mem':<12} | {'Speedup':<10}")
        print("-" * 140)
        
        results = {}
        shape = (batch_size * seqlen, nheads, headdim)
        q = torch.randn(shape, device=device, dtype=dtype)
        k = torch.randn(shape, device=device, dtype=dtype)
        v = torch.randn(shape, device=device, dtype=dtype)
        cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=device)
        
        q_sdpa = q.reshape(batch_size, seqlen, nheads, headdim).transpose(1, 2).contiguous()
        k_sdpa = k.reshape(batch_size, seqlen, nheads, headdim).transpose(1, 2).contiguous()
        v_sdpa = v.reshape(batch_size, seqlen, nheads, headdim).transpose(1, 2).contiguous()

        # Benchmark Base (Torch SDPA)
        base_f = time_fwd(
            F.scaled_dot_product_attention,
            q_sdpa,
            k_sdpa,
            v_sdpa,
            dropout_p=dropout_p,
            is_causal=causal,
            repeats=repeats,
            verbose=False,
        )
        base_mem_mb = measure_peak_memory_mb(
            F.scaled_dot_product_attention,
            q_sdpa,
            k_sdpa,
            v_sdpa,
            dropout_p=dropout_p,
            is_causal=causal,
        )
        base_speed = efficiency(flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"), base_f)
        
        results["base"] = [[base_f], [base_speed], [base_mem_mb]]
        
        sparsity_list = get_sparsity_list(sparsity_sampling_steps, seqlen, causal)
        
        for sparsity in sparsity_list:
            sum_sparsity, sum_speed, sum_latency, sum_mem = 0, 0, 0, 0
            
            for _ in range(block_sparse_repeats):
                # Re-generate mask for each repeat
                base_blockmask, real_sparsity = generate_base_sparsity_mask(
                    seqlen, seqlen, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, 
                    sparsity, causal=causal, device=device
                )
                base_blockmask = base_blockmask.unsqueeze(0).repeat(batch_size, nheads, 1, 1)
                head_mask_type = torch.tensor([1] * nheads, device=device, dtype=torch.int32)
                
                f = time_fwd(
                    block_sparse_attn_func, q, k, v, cu_seqlens, cu_seqlens, 
                    head_mask_type, None, base_blockmask, seqlen, seqlen, 
                    dropout_p, is_causal=causal, exact_streaming=False,
                    # m_block_dim=BLOCK_SIZE, n_block_dim=BLOCK_SIZE,
                    repeats=repeats, verbose=False
                )
                mem_mb = measure_peak_memory_mb(
                    block_sparse_attn_func, q, k, v, cu_seqlens, cu_seqlens, 
                    head_mask_type, None, base_blockmask, seqlen, seqlen, 
                    dropout_p, is_causal=causal, exact_streaming=False,
                    # m_block_dim=BLOCK_SIZE, n_block_dim=BLOCK_SIZE,
                )
                
                speed = efficiency(flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"), f)
                
                sum_sparsity += real_sparsity
                sum_speed += speed
                sum_latency += f
                sum_mem += mem_mb

            avg_sparsity = sum_sparsity / block_sparse_repeats
            avg_speed = sum_speed / block_sparse_repeats
            avg_latency = sum_latency / block_sparse_repeats
            avg_mem_mb = sum_mem / block_sparse_repeats
            speedup = base_f / avg_latency if avg_latency > 0 else 0.0

            if avg_sparsity not in results:
                results[avg_sparsity] = [[], [], []]
            results[avg_sparsity][0].append(avg_latency)
            results[avg_sparsity][1].append(avg_speed)
            results[avg_sparsity][2].append(avg_mem_mb)
            
            excel_data.append([batch_size, seqlen, avg_sparsity, avg_speed, avg_latency, avg_mem_mb, speedup, base_speed, base_f, base_mem_mb])
            
            print(f"{avg_sparsity:<10.4f} | {base_f*1000:>12.2f} ms | {base_speed:>12.2f} | {base_mem_mb:>9.1f} MB | {avg_latency*1000:>12.2f} ms | {avg_speed:>12.2f} | {avg_mem_mb:>9.1f} MB | {speedup:>9.2f}x")

        # Summarize results for JSON
        final_results = {}
        for key, val in results.items():
            if key == "base":
                final_results[key] = [val[0][0], val[1][0], val[2][0]]
            else:
                final_results[key] = [
                    sum(val[0]) / len(val[0]),
                    sum(val[1]) / len(val[1]),
                    sum(val[2]) / len(val[2]),
                ]
        all_results[seqlen] = final_results

    # Save outputs
    with open(f"all_results_{excel_file_name}.json", "w") as f:
        json.dump(all_results, f, indent=4)
            
    write_to_excel(excel_label, excel_data, excel_dir_path, excel_file_name)
    print(f"\nResults saved to {excel_dir_path}{excel_file_name}.xlsx and all_results_{excel_file_name}.json")

if __name__ == "__main__":
    profile_blocksparse_fwd()
