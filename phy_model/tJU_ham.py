import torch
import numpy as np
from utils.parity_perserving_linalg.parity_preserving_expm import parity_preserving_expm

def build_ham(t, J, U, mu, h, hs, z, dtype=torch.float64, device='cpu'):
    # """Hubbard Hamiltonian
    #     H = -t c_{i,\sigma}^\dagger c_{j,\sigma} + U n_{i,\uparrow} n_{i,\downarrow} - mu (n_{i,\uparrow} + n_{i,\downarrow}) + h ( n_{i,\uparrow} - n_{i,\downarrow} )
    # """
    # @ half filling, \mu = U/2
    
    parity = [1,1,-1,-1]
    
    H = torch.zeros(4,4,4,4, dtype=dtype, device=device)
    
    #    0,     1,   2,    3
    # [|0>, |u,d>, |u>, |d>]
    
    H[0,2,2,0], H[0,1,2,3], H[2,0,0,2], H[2,3,0,1] = -t, -t, -t, -t
    H[0,3,3,0], H[2,3,1,0], H[3,0,0,3], H[1,0,2,3] = -t, -t, -t, -t
    H[3,2,1,0], H[3,1,1,3], H[1,0,3,2], H[1,3,3,1] = t, t, t, t
    H[0,1,3,2], H[2,1,1,2], H[3,2,0,1], H[1,2,2,1] = t, t, t, t
    
    # ======================================
    H[0,0,0,0] = 0
    H[1,1,1,1] = 2*(U-2*mu)/z
    H[0,1,0,1] = H[1,0,1,0] = (U-2*mu)/z
    H[0,2,0,2] = H[0,3,0,3] = H[2,0,2,0] = H[3,0,3,0] = -mu/z
    H[1,2,1,2] = H[1,3,1,3] = H[2,1,2,1] = H[3,1,3,1] = (U-3*mu)/z
    H[2,3,2,3] = H[3,2,3,2] = H[2,2,2,2] = H[3,3,3,3] = -2*mu/z
    
    # ======================================
    H[0,2,0,2] += h/2/z
    H[1,2,1,2] += h/2/z
    H[2,0,2,0] += h/2/z
    H[2,1,2,1] += h/2/z
    
    H[0,3,0,3] -= h/2/z
    H[1,3,1,3] -= h/2/z
    H[3,0,3,0] -= h/2/z
    H[3,1,3,1] -= h/2/z

    H[2,2,2,2] += +h/z
    H[3,3,3,3] += -h/z
    
    # ======================================
    H[0,2,0,2] += -hs/2/z
    H[1,2,1,2] += -hs/2/z
    H[2,0,2,0] += +hs/2/z
    H[2,1,2,1] += +hs/2/z
    
    H[0,3,0,3] -= -hs/2/z
    H[1,3,1,3] -= -hs/2/z
    H[3,0,3,0] -= +hs/2/z
    H[3,1,3,1] -= +hs/2/z
    
    H[2,3,2,3] += +hs/z
    H[3,2,3,2] += -hs/z
    
    #======================================
    
    # J S_i^z S_j^z
    H[2,2,2,2] += +J/4
    H[2,3,2,3] += -J/4
    H[3,2,3,2] += -J/4
    H[3,3,3,3] += +J/4
    
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



def build_gate(tau, t, J, U, mu, h, hs, z, dtype=torch.float64, device='cpu'):
    
    ham = build_ham(t=t, J=J, U=U, mu=mu, h=h, hs=hs, z=z, dtype=dtype, device=device)
    
    ham_mat_coeff = ham.permute(0,1,3,2).reshape(16, 16)
    
    p_row = np.kron([-1,-1,1,1],[-1,-1,1,1])
    p_col = np.kron([-1,-1,1,1],[-1,-1,1,1])
    
    expm = parity_preserving_expm(ham_mat_coeff, tau, p_row, p_col)
    gate = expm.reshape(4,4,4,4).permute(0,1,3,2)
    
    return gate
    