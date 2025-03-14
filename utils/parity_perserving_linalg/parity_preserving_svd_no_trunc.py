import torch
import numpy as np
from textwrap import dedent
import matplotlib.pyplot as plt
# from dask.distributed import Client

def calc_order_inv(order:np.array):
    order_inv = np.zeros_like(order, dtype=np.int64)
    order_inv[order] = np.arange(len(order), dtype=np.int64)
    return order_inv

def check_block_diag_form(mat, block_dim_row, block_dim_col):
    mat_recover = torch.block_diag(mat[:block_dim_row,:block_dim_col],mat[block_dim_row:,block_dim_col:])
    flag = torch.allclose(mat, mat_recover)
    diff = torch.linalg.norm(mat - mat_recover)
    if not flag:
        print(f"diff = {diff}")
        plt.imshow(mat)
        plt.savefig("ill-mat.png", dpi=300)
    return flag



def parity_preserving_svd_no_truncation(mat, parity_row_and_col:np.array):

    odd_parity_num = np.sum(parity_row_and_col<0)    
    
    sorted_p_inds = np.argsort(parity_row_and_col)
    
    sorted_p_inv = calc_order_inv(sorted_p_inds)
        
    mat = mat[sorted_p_inds,:]
    mat = mat[:,sorted_p_inds]
        
    mat_p_odd = mat[:odd_parity_num, :odd_parity_num]
    mat_p_even = mat[odd_parity_num:, odd_parity_num:]
    
    u_odd, s_odd, vh_odd = torch.linalg.svd(mat_p_odd)
    u_even, s_even, vh_even = torch.linalg.svd(mat_p_even)
    
    s_combined = torch.cat([s_odd, s_even])
    u_combined = torch.block_diag(u_odd, u_even)
    vh_combined = torch.block_diag(vh_odd, vh_even)

    s_combined = s_combined[sorted_p_inv]
    
    u_combined = u_combined[:,sorted_p_inv]
    u_combined = u_combined[sorted_p_inv,:]
    
    vh_combined = vh_combined[:,sorted_p_inv]
    vh_combined = vh_combined[sorted_p_inv,:]
    
    
    return u_combined, s_combined, vh_combined