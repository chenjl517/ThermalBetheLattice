import torch
import numpy as np
from textwrap import dedent
import matplotlib.pyplot as plt

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

def lq_decomposition(mat):
    matT = mat.conj().T
    q, r = torch.linalg.qr(matT)
    
    L_prime = r.conj().T
    Q_prime = q.conj().T
    
    return L_prime, Q_prime


def parity_preserving_qr(mat, parity_row:np.array, parity_col:np.array, method='qr'):

    
    odd_parity_num_row = np.sum(parity_row<0)
    odd_parity_num_col = np.sum(parity_col<0)
    
    sorted_p_row_inds = np.argsort(parity_row)
    sorted_p_col_inds = np.argsort(parity_col)
    
    sorted_p_row_inv = calc_order_inv(sorted_p_row_inds)
    sorted_p_col_inv = calc_order_inv(sorted_p_col_inds)
    
    mat = mat[sorted_p_row_inds,:]
    mat = mat[:,sorted_p_col_inds]
    
    ############################# check ################################
    try:
        assert check_block_diag_form(mat, odd_parity_num_row, odd_parity_num_col), "The input matrix is not in block diagonal form"
    except Exception as e:
        print(parity_row)
        print(parity_col)
        print(mat.abs().max())
        mat10 = mat[odd_parity_num_row:, :odd_parity_num_col]
        mat01 = mat[:odd_parity_num_row, odd_parity_num_col:]
        # get the max value pos
        pos10 = torch.argmax(mat10.abs())
        pos01 = torch.argmax(mat01.abs())
        print("row = ", odd_parity_num_row, "col = ", odd_parity_num_col)
        print(mat10.abs().max(), pos10)
        print(mat01.abs().max(), pos01)
        raise e
    ####################################################################
    
    # print(f"block mat = ")
    # print(mat)
        
    mat_p_odd = mat[:odd_parity_num_row, :odd_parity_num_col]
    mat_p_even = mat[odd_parity_num_row:, odd_parity_num_col:]
    
    if method == 'qr':
        q_odd, r_odd   = torch.linalg.qr(mat_p_odd)
        q_even, r_even = torch.linalg.qr(mat_p_even)
    elif method == 'lq':
        q_odd, r_odd   = lq_decomposition(mat_p_odd)
        q_even, r_even = lq_decomposition(mat_p_even)
    else:
        raise ValueError(f"Invalid method: {method}")
    
    Q_mat = torch.block_diag(q_odd, q_even)
    R_mat = torch.block_diag(r_odd, r_even)
    
    Q_mat = Q_mat[sorted_p_row_inv,:]
    R_mat = R_mat[:,sorted_p_col_inv]
    parity = [-1]*q_odd.shape[1] + [1]*q_even.shape[1]
    
    # print(f"mat.shape = {mat.shape}, Q.shape = {Q_mat.shape}, R.shape = {R_mat.shape}")
    
    return Q_mat, R_mat, parity

    
if __name__ == "__main__":
    from parity_preserving_expm import parity_preserving_expm
    
    def build_ham(t, dtype=torch.float64, device='cpu'):
        """Heisenberg Chain Hamiltonian
        
        H = -t c_i^\dagger c_{i+1} + h.c.
        """
        
        Hamiltonian = torch.zeros(2,2,2,2, dtype=dtype, device=device)
        
        Hamiltonian[1,0,1,0] = -t
        Hamiltonian[0,1,0,1] = -t

        return Hamiltonian

        
    tau     = 0.1
    t       = 1.0
    
    ham     = build_ham(t)
    
    p_row = np.kron([-1,1],[-1,1])
    p_col = np.kron([-1,1],[-1,1])
    
    ham_normal_order = np.transpose(ham,(0,1,3,2)).reshape(4,4)
    print(ham_normal_order)
    
    
    Q, R, parity = parity_preserving_qr(ham.reshape(4,4),  p_row, p_col, method='qr')
    
    print(f"Q = \n{Q}")
    print(f"R = \n{R}")
    print(f"parity = {parity}")
    