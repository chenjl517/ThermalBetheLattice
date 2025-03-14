import torch
import numpy as np
from utils.parity_perserving_linalg.utils.parity_preserving_expm import parity_preserving_expm


def build_ham(J, Delta, z, h, dtype=torch.float64, device='cpu'):
    
    parity = [1,1]
    
    s_z = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device) / 2
    s_x = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device) / 2
    s_y_reduce = torch.tensor([[0, -1], [1, 0]], dtype=dtype, device=device) / 2   
    Identity = torch.eye(2, dtype=dtype, device=device)
    
    H = J * (torch.kron(s_x, s_x) - torch.kron(s_y_reduce, s_y_reduce)) + Delta * torch.kron(s_z, s_z)
    H += h * (torch.kron(s_z, Identity) + torch.kron(Identity, s_z))/z
    H = H.reshape(2,2,2,2)
    
    
    # .reshape(2,2,2,2)
    # H = -J*torch.kron(sigma_z, sigma_z).reshape(2,2,2,2)
    
    # label permutation
    # 0  1
    # 2  3
    H = H.permute(1,0,2,3)
    
    # |u>, |d>, |0>, |ud>
    sorted_parity = np.argsort(parity)
    
    H = H[sorted_parity, :, :, :]
    H = H[:, sorted_parity, :, :]
    H = H[:, :, sorted_parity, :]
    H = H[:, :, :, sorted_parity]

    return H



def build_gate(tau, J, Delta, z, h, dtype=torch.float64, device='cpu'):
    
    ham = build_ham(J=J, Delta=Delta, z=z, h=h, dtype=dtype, device=device)
    
    ham_mat_coeff = ham.permute(0,1,3,2).reshape(4, 4)
    
    
    p_row = np.kron([1,1],[1,1])
    p_col = np.kron([1,1],[1,1])
    
    expm = parity_preserving_expm(ham_mat_coeff, tau, p_row, p_col)
    gate = expm.reshape(2,2,2,2).permute(0,1,3,2)
    
    return gate
    