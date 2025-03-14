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


def parity_preserving_expm(mat, tau, parity_row:np.array, parity_col:np.array):

    
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
    
    w_odd,v_odd = torch.linalg.eigh(mat_p_odd)
    w_even,v_even = torch.linalg.eigh(mat_p_even)
    
    exp_mat_odd  = v_odd  @ torch.diag(torch.exp(-tau*w_odd))  @ v_odd.T.conj()
    exp_mat_even = v_even @ torch.diag(torch.exp(-tau*w_even)) @ v_even.T.conj()
    
    exp_mat = torch.block_diag(exp_mat_odd, exp_mat_even)
    
    # print(f"block mat = ")
    # print(exp_mat)
    
    exp_mat = exp_mat[sorted_p_row_inv,:]
    exp_mat = exp_mat[:,sorted_p_col_inv]
    
    return exp_mat    
    
    
if __name__ == "__main__":
    
    def build_ham(t, dtype=torch.float64, device='cpu'):
        """Heisenberg Chain Hamiltonian
        
        H = -t c_i^\dagger c_{i+1} + h.c.
        """
        
        Hamiltonian = torch.zeros(2,2,2,2, dtype=dtype, device=device)
        
        Hamiltonian[1,0,1,0] = -t
        Hamiltonian[0,1,0,1] = -t

        return Hamiltonian


    def build_gate(tau, t, dtype=torch.float64, device='cpu'):
        gate = torch.zeros(2,2,2,2,dtype=dtype, device=device)
        
        # |1>, |0>
        gate[1,1,1,1] = 1
        gate[0,1,1,0] = 1
        gate[1,0,0,1] = 1
        gate[0,0,0,0] = 1 + (t*tau)**2
        gate[1,0,1,0] = t*tau
        gate[0,1,0,1] = t*tau
        
        return gate
    
    def build_gate2(tau, t, dtype=torch.float64, device='cpu'):
        gate = torch.zeros(2,2,2,2,dtype=dtype, device=device)
        # |0>, |1>
        
        Sp       = torch.tensor([[0, 1], [0, 0]], dtype=dtype, device=device)
        Sm       = torch.tensor([[0, 0], [1, 0]], dtype=dtype, device=device)
        
        Sp = Sp[:,[1,0]]
        Sp = Sp[[1,0],:]
        Sm = Sm[:,[1,0]]
        Sm = Sm[[1,0],:]
        
        Hamiltonian = -t*(torch.kron(Sp, Sm) + torch.kron(Sm, Sp))
        
        gate    = torch.linalg.matrix_exp(-tau*Hamiltonian)
        gate    = gate.view(2,2,2,2).permute(1,0,2,3)
        
        # |1>, |0>
        # gate[1,1,1,1] = 1
        # gate[0,1,1,0] = 1
        # gate[1,0,0,1] = 1
        # gate[0,0,0,0] = 1 + (t*tau)**2
        # gate[1,0,1,0] = t*tau
        # gate[0,1,0,1] = t*tau
        
        return gate

        
    tau     = 0.1
    t       = 1.0
    
    ham     = build_ham(t)
    
    p_row = np.kron([-1,1],[-1,1])
    p_col = np.kron([-1,1],[-1,1])
    
    ham_normal_order = np.transpose(ham,(0,1,3,2)).reshape(4,4)
    print(ham_normal_order)
    
    expm_op = np.transpose(parity_preserving_expm(np.transpose(ham,(0,1,3,2)).reshape(4,4), tau, p_row, p_col).reshape(2,2,2,2),(0,1,3,2))
    
    gate_exact = build_gate(tau, t)
    gate_exact2 = build_gate2(tau, t)
    
    print("expm_op\n",expm_op.reshape(4,4))
    print("exact\n",gate_exact.reshape(4,4))
    print("exact2\n",gate_exact2.reshape(4,4))
    print(np.allclose(expm_op, gate_exact))