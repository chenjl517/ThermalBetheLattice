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



def parity_preserving_eigh(mat, parity_row_and_col:np.array):

    odd_parity_num = np.sum(parity_row_and_col<0)    
    
    sorted_p_inds = np.argsort(parity_row_and_col)
    
    sorted_p_inv = calc_order_inv(sorted_p_inds)
        
    mat = mat[sorted_p_inds,:]
    mat = mat[:,sorted_p_inds]
        
    mat_p_odd = mat[:odd_parity_num, :odd_parity_num]
    mat_p_even = mat[odd_parity_num:, odd_parity_num:]
    
    w_odd, v_odd   = torch.linalg.eigh(mat_p_odd)
    w_even, v_even = torch.linalg.eigh(mat_p_even)
    
    w_combined = torch.cat([w_odd, w_even])
    v_combined = torch.block_diag(v_odd, v_even)

    w_combined = w_combined[sorted_p_inv]
    v_combined = v_combined[:,sorted_p_inv]
    v_combined = v_combined[sorted_p_inv,:]
    
    return w_combined, v_combined