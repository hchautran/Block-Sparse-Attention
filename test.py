                                                                                                                                       
                                                                                                                                                           
# from block_sparse_attn import block_sparse_attn_func                                                                                                     
from examples.vit_masks import generate_vit_spatial_locality_mask                                                                                        
                                                                                                                                                        
# 1. Generate mask                                                                                                                                       
mask = generate_vit_spatial_locality_mask(                                                                                                               
    img_size=224, patch_size=16, block_size=128,                                                                                                         
    batch_size=4, num_heads=12, locality_radius=1                                                                                                        
)                                                                                                                                                        
print(mask)
                                                                                                                                                        
# # 2. Use with your attention                                                                                                                             
# output = block_sparse_attn_func(                                                                                                                         
#     q_unpad, k_unpad, v_unpad,                                                                                                                           
#     cu_seqlens_q, cu_seqlens_k,                                                                                                                          
#     head_mask_type, None, mask,                                                                                                                          
#     max_seqlen_q, max_seqlen_k,                                                                                                                          
#     p_dropout=0.0, is_causal=False                                                                                                                       
# )   