import torch
import os
import numpy as np
import opt_einsum as oe
from utils.seq_kron import seq_kron

from utils.parity_perserving_linalg.parity_preserving_svd import parity_preserving_svd
from utils.parity_perserving_linalg.parity_preserving_qr import parity_preserving_qr
from utils.parity_perserving_linalg.parity_preserving_eigh import parity_preserving_eigh
from utils.parity_perserving_linalg.parity_preserving_svd_no_trunc import parity_preserving_svd_no_truncation

from utils.high_order_trotter import get_scheme_coeff

from copy import deepcopy

import itertools
import json

def safe_inv(lam, threshold=1e-20):
    return 1/(lam + threshold)


pretty_print = lambda x: print(torch.round(x*1e6)/1e6)

class BetheLattice:
    def __init__(self, z, 
                 coupling_x, coupling_y, coupling_z,
                 gate, ham,
                 ham_basis_parity,
                 chemical_potential, 
                 chi_max, 
                 ignore_threshold=1e-7, 
                 statistic="Fermion", 
                 calc_mode = 'thermal',
                 dtype=torch.float64, device='cpu',
                 measure=False):


        self.lnZ = 0
        self.measure = measure
        
        if statistic == "Fermion":
            self.swap_on = -1
        elif statistic == "Boson":
            self.swap_on = 1
        else:
            raise ValueError("statistic must be 'Fermion' or 'Boson'")
        
        self.z = z
        
        self.beta = 0
        self.evolve_step = 0
        
        self.chi_max = chi_max
        self.ignore_threshold = ignore_threshold
        self.dtype = dtype
        self.device = device

        self.calc_mode = calc_mode
        
        self._model_initialized_flag = False
        
        self.parity_dtype  = np.int8

        self.parity_tabel = {}
        
        self.set_model(ham, gate, coupling_x, coupling_y, coupling_z, ham_basis_parity)
        
        self._initialize_parity_tabel()
        self._initialize_tensor()
        
        self.obs_list = []

        with open(os.path.join("contraction_rule", f"contraction_rule_z={self.z}.json"), 'r') as f:
            self.contraction_rule = json.load(f)
                
        self.dask_client = None
        
        self.mu_max = 10
        
        self.chemical_potential = chemical_potential
        self.summation_beta_mu  = 0
        
    def _initialize_tensor(self):
        
        T_shape = [1] * (self.z) + [self.local_hilbert_space_dim_1, self.local_hilbert_space_dim_2]
        
        if self.calc_mode == "thermal":
            
            self.T1 = torch.eye(self.local_hilbert_space_dim_1, dtype=self.dtype, device=self.device)
            self.T2 = torch.eye(self.local_hilbert_space_dim_1, dtype=self.dtype, device=self.device)
            
            self.T1 = self.T1.reshape(T_shape)
            self.T2 = self.T2.reshape(T_shape)
        else:
            self.T1 = torch.ones(self.local_hilbert_space_dim_1, self.local_hilbert_space_dim_2, dtype=self.dtype, device=self.device)
            self.T2 = torch.ones(self.local_hilbert_space_dim_1, self.local_hilbert_space_dim_2, dtype=self.dtype, device=self.device)
            
            self.T1 = self.T1.reshape(T_shape)
            self.T2 = self.T2.reshape(T_shape)
            
            self._clean_tensor_parity()
        
        self.lam = {}
        for i in range(self.z):
            self.lam[i] = torch.ones(1, dtype=self.dtype, device=self.device)
    
    def _initialize_parity_tabel(self):
        for i in range(self.z):
            self.parity_tabel[i] = np.array([1],dtype=self.parity_dtype)
        print("parity_tabel = ", self.parity_tabel)
        
    def _clean_tensor_parity(self):
        for tensor in [self.T1, self.T2]:
            for index in itertools.product(*[range(i) for i in tensor.shape]):
                tol_p = 1
                for ii in range(self.z+2):
                    tol_p *= self.parity_tabel[ii][index[ii]]                                            
                if tol_p == -1:
                    print(f"tensor[{index}] = {tensor[index]}")
                    print(f"parity not preserved at ({index}), FORCE TO ZERO")
                    tensor[index] = 0

    
    def apply_gate_and_update_i(self, bond_idx, gate=None):

        # rotate the tensor to the normal order
        contraction_rule_forward_1 = self.contraction_rule["permutation_rule"][f"{bond_idx}"]["1_forward"]
        crossing_legs_1 = self.contraction_rule["permutation_rule"][f"{bond_idx}"]["1_crossing"]
        outer_legs_1 = self.contraction_rule["permutation_rule"][f"{bond_idx}"]["1_outer"]
        
        swap_tensor_list_1 = []
        for cross_i in crossing_legs_1:
            swap_i = torch.ones(len(self.parity_tabel[self.z]), len(self.parity_tabel[cross_i]), dtype=self.dtype, device=self.device)
            minus_parity_phy1 = np.sum(self.parity_tabel[self.z]==-1)
            minus_parity_cross_i = np.sum(self.parity_tabel[cross_i]==-1)
            swap_i[:minus_parity_phy1, :minus_parity_cross_i] = self.swap_on
            swap_tensor_list_1.append(swap_i)
        
        outer_tensor_list_1 = [self.lam[i] for i in outer_legs_1]
        
        T1_prime = oe.contract(contraction_rule_forward_1, self.T1, *swap_tensor_list_1, *outer_tensor_list_1)

        ######################

        contraction_rule_forward_2 = self.contraction_rule["permutation_rule"][f"{bond_idx}"]["2_forward"]
        crossing_legs_2 = self.contraction_rule["permutation_rule"][f"{bond_idx}"]["2_crossing"]
        outer_legs_2 = self.contraction_rule["permutation_rule"][f"{bond_idx}"]["2_outer"]
        
        swap_tensor_list_2 = []
        for cross_i in crossing_legs_2:
            swap_i = torch.ones(len(self.parity_tabel[self.z]), len(self.parity_tabel[cross_i]), dtype=self.dtype, device=self.device)
            minus_parity_phy1 = np.sum(self.parity_tabel[self.z]==-1)
            minus_parity_cross_i = np.sum(self.parity_tabel[cross_i]==-1)
            swap_i[:minus_parity_phy1, :minus_parity_cross_i] = self.swap_on
            swap_tensor_list_2.append(swap_i)
        
        outer_tensor_list_2 = [self.lam[i] for i in outer_legs_2]
        
        T2_prime = oe.contract(contraction_rule_forward_2, self.T2, *swap_tensor_list_2, *outer_tensor_list_2)

        # --------------------------------------------------------------------------------------------
        # APPLY GATE
        # --------------------------------------------------------------------------------------------
        permuted_indices_1 = self.contraction_rule["permutation_rule"][f"{bond_idx}"]["permuted_indices_1"]
        T1_row_parity_list = [self.parity_tabel[permuted_indices_1[i]] for i in range(self.z)]
        T1_col_paritiy_list = [self.parity_tabel[permuted_indices_1[self.z]], self.parity_tabel[permuted_indices_1[self.z+1]]]
        T1_prow = seq_kron(*T1_row_parity_list)
        T1_pcol = seq_kron(*T1_col_paritiy_list)
        T1_mat  = T1_prime.reshape(len(T1_prow), len(T1_pcol))
        Q1, R1, intermidiate_parity_L = parity_preserving_qr(T1_mat, T1_prow, T1_pcol, method='qr')
        R1 = R1.reshape(len(intermidiate_parity_L), T1_prime.shape[self.z], T1_prime.shape[self.z+1])

        permuted_indices_2 = self.contraction_rule["permutation_rule"][f"{bond_idx}"]["permuted_indices_2"]        
        T2_row_parity_list = [self.parity_tabel[permuted_indices_2[0]], self.parity_tabel[permuted_indices_2[1]]]
        T2_col_paritiy_list = [self.parity_tabel[permuted_indices_2[i]] for i in range(2, self.z+2)]
        T2_prow = seq_kron(*T2_row_parity_list)
        T2_pcol = seq_kron(*T2_col_paritiy_list)
        T2_mat  = T2_prime.reshape(len(T2_prow), len(T2_pcol))
        L2,Q2,intermidiate_parity_R = parity_preserving_qr(T2_mat, T2_prow, T2_pcol, method='lq')
        L2 = L2.reshape(T2_prime.shape[0], T2_prime.shape[1], len(intermidiate_parity_R))
        
        if gate is None:
            theta = oe.contract("jce, e, efk -> jc fk", R1, self.lam[bond_idx], L2)
        else:
            theta = oe.contract("jce, e, efk, fcml -> jm lk", R1, self.lam[bond_idx], L2, gate)
                    
        theta_prow  = seq_kron(intermidiate_parity_L, self.parity_tabel[permuted_indices_1[self.z]])
        theta_pcol  = seq_kron(self.parity_tabel[permuted_indices_2[1]], intermidiate_parity_R)
        theta_mat   = theta.reshape(len(theta_prow), len(theta_pcol))
        
        u, updated_center_bond, vh, updated_center_bond_parity, cutoff = parity_preserving_svd(theta_mat, theta_prow, theta_pcol, chi_max=self.chi_max, singular_threshold=self.ignore_threshold, force_keep=False, verbose=False, dask_client=self.dask_client)
        
        u  = u.reshape(len(intermidiate_parity_L), self.local_hilbert_space_dim_1, len(updated_center_bond_parity))
        vh = vh.reshape(len(updated_center_bond_parity), self.local_hilbert_space_dim_1, len(intermidiate_parity_R))
        
        # --------------------------------------------------------------------------------------------
        # RECOVER & UPDATE TENSOR
        # --------------------------------------------------------------------------------------------
        # update bond tensor
        coeff = updated_center_bond.norm()
        updated_center_bond = updated_center_bond / coeff
        self.lnZ += torch.log(coeff) # NO `/2` for we purificarion \rho(\beta/2) \rho(\beta/2)^\dagger
        
        self.lam[bond_idx] = updated_center_bond
        self.parity_tabel[bond_idx] = updated_center_bond_parity
        
        
        # update T1
        T1_prime_prime = oe.contract("ab, bcd -> acd", Q1, u)
        T1_prime_shape_new = [len(self.parity_tabel[i]) for i in permuted_indices_1]
        
        T1_prime       = T1_prime_prime.reshape(T1_prime_shape_new)

        contraction_rule_backward_1 = self.contraction_rule["permutation_rule"][f"{bond_idx}"]["1_backward"]
        swap_tensor_list_1 = []
        for cross_i in crossing_legs_1:
            swap_i = torch.ones(len(self.parity_tabel[self.z]), len(self.parity_tabel[cross_i]), dtype=self.dtype, device=self.device)
            minus_parity_phy1 = np.sum(self.parity_tabel[self.z]==-1)
            minus_parity_cross_i = np.sum(self.parity_tabel[cross_i]==-1)
            swap_i[:minus_parity_phy1, :minus_parity_cross_i] = self.swap_on
            swap_tensor_list_1.append(swap_i)
        
        outer_tensor_inv_list_1 = [safe_inv(self.lam[i]) for i in outer_legs_1]
        
        self.T1 = oe.contract(contraction_rule_backward_1, T1_prime, *swap_tensor_list_1, *outer_tensor_inv_list_1)

        # update T2
        T2_prime_prime = oe.contract("efg,gh->efh", vh, Q2)
        T2_prime_shape_new = [len(self.parity_tabel[i]) for i in permuted_indices_2]
        
        T2_prime       = T2_prime_prime.reshape(T2_prime_shape_new)

        contraction_rule_backward_2 = self.contraction_rule["permutation_rule"][f"{bond_idx}"]["2_backward"]
        swap_tensor_list_2 = []
        for cross_i in crossing_legs_2:
            swap_i = torch.ones(len(self.parity_tabel[self.z]), len(self.parity_tabel[cross_i]), dtype=self.dtype, device=self.device)
            minus_parity_phy1 = np.sum(self.parity_tabel[self.z]==-1)
            minus_parity_cross_i = np.sum(self.parity_tabel[cross_i]==-1)
            swap_i[:minus_parity_phy1, :minus_parity_cross_i] = self.swap_on
            swap_tensor_list_2.append(swap_i)
        
        outer_tensor_inv_list_2 = [safe_inv(self.lam[i]) for i in outer_legs_2]
        
        self.T2 = oe.contract(contraction_rule_backward_2, T2_prime, *swap_tensor_list_2, *outer_tensor_inv_list_2)
        
        return cutoff


    def set_model(self, model_ham, model_gate, coupling_x, coupling_y, coupling_z, local_hilbert_space_parity):
        self.model_ham = model_ham
        self.model_gate = model_gate
        
        assert coupling_x == coupling_y == coupling_z
        
        self.coupling = [coupling_x for _ in range(self.z)]
        
        self.parity_tabel[self.z]  = np.array(local_hilbert_space_parity, dtype=self.parity_dtype)
        if self.calc_mode == "thermal":
            self.parity_tabel[self.z+1] = np.array(local_hilbert_space_parity, dtype=self.parity_dtype)
        elif self.calc_mode == "gs":
            self.parity_tabel[self.z+1] = np.array([1], dtype=self.parity_dtype)
            

        self.local_hilbert_space_dim_1 = len(self.parity_tabel[self.z])
        self.local_hilbert_space_dim_2 = len(self.parity_tabel[self.z+1])
        
        self.phy_swap = torch.ones(self.local_hilbert_space_dim_2, self.local_hilbert_space_dim_1, dtype=self.dtype, device=self.device)
        
        minus_parity_phy1 = np.sum(self.parity_tabel[self.z]==-1)
        minus_parity_phy2 = np.sum(self.parity_tabel[self.z+1]==-1)
        
        self.phy_swap[:minus_parity_phy1, :minus_parity_phy2] = self.swap_on
        
        print("phy swap = ", self.phy_swap)
        
        
        if self.swap_on == -1:
            # # |u>, |d>, |0>, |ud>
            self.n_up = torch.zeros(self.local_hilbert_space_dim_1, self.local_hilbert_space_dim_1,  dtype=self.dtype, device=self.device)
            self.n_up[0,0] = 1
            self.n_up[3,3] = 1
            
            self.n_down = torch.zeros(self.local_hilbert_space_dim_1, self.local_hilbert_space_dim_1,  dtype=self.dtype, device=self.device)
            self.n_down[1,1] = 1
            self.n_down[3,3] = 1
            
            self.sz = torch.zeros(self.local_hilbert_space_dim_1, self.local_hilbert_space_dim_1,  dtype=self.dtype, device=self.device)
            self.sz[0,0] = 1
            self.sz[1,1] = -1
            
            self.n_double_occ = torch.zeros(self.local_hilbert_space_dim_1, self.local_hilbert_space_dim_1,  dtype=self.dtype, device=self.device)
            self.n_double_occ[3,3] = 1
            
            self.sx = torch.zeros(self.local_hilbert_space_dim_1, self.local_hilbert_space_dim_1, dtype=self.dtype, device=self.device)
            self.sx[0,1] = 1
            self.sx[1,0] = 1
            
            self.n_tot = self.n_up + self.n_down

    def canonicalize(self, cano_eps=1e-9, eigen_eps=1e-7, maxiter=200, cano_scheme="simple"):
        if self.calc_mode == "gs":
            # print("Canonicalization is off in ground state calculation.")
            return
        for i in range(maxiter):

            if cano_scheme == "simple":
                for _ in range(50):
                    for i in range(self.z):    
                        self.apply_gate_and_update_i(i, gate = None)
            else:
                self.fix_to_canonical_gauge(matvec_maxiter=100, eigen_eps=eigen_eps)
            
            if self.verify_canonical(threshold=cano_eps, verbose=False):
                break    
        

    def evolve_once(self, evolve_step, cano_scheme="simple"):
        
        cutoff_list = []
        for bond_idx in range(self.z):
            cutoff = self.apply_gate_and_update_i(bond_idx, gate=self.model_gate(evolve_step, self.coupling[bond_idx]))
            cutoff_list.append(cutoff)
            self.canonicalize(cano_eps=1e-10, eigen_eps=1e-14, maxiter = 50, cano_scheme=cano_scheme)
                
        

    def evolve_once_v2(self, evolve_step, cano_scheme="simple", trotter_scheme="2"):
        # print(f"Trotter Scheme = {trotter_scheme}")
        if trotter_scheme == "1":
            return self.evolve_once(evolve_step, cano_scheme=cano_scheme)
        else:
            cutoff_list = []
            ci_list, di_list = get_scheme_coeff(trotter_scheme)
            for idx in range(len(ci_list)):

                ci = ci_list[idx]
                di = di_list[idx]
                
                for bond_idx in range(self.z-1):
                    print(f"evolve bond {bond_idx}... | ci = {ci}")
                    cutoff = self.apply_gate_and_update_i(bond_idx, gate=self.model_gate(evolve_step*ci, self.coupling[bond_idx]))
                    cutoff_list.append(cutoff)
                    print(f"cutoff = {cutoff}")
                    self.canonicalize(cano_eps=1e-10, eigen_eps=1e-14, maxiter = 50, cano_scheme=cano_scheme)
                
                cutoff = self.apply_gate_and_update_i(self.z-1, gate=self.model_gate(evolve_step*(ci+di), self.coupling[self.z-1]))    
                cutoff_list.append(cutoff)
                print(f"cutoff = {cutoff}")
                self.canonicalize(cano_eps=1e-10, eigen_eps=1e-14, maxiter = 50, cano_scheme=cano_scheme)
                
                for bond_idx in range(self.z-2, -1, -1):
                    print(f"evolve bond {bond_idx}... | di = {di}")
                    cutoff = self.apply_gate_and_update_i(bond_idx, gate=self.model_gate(evolve_step*di, self.coupling[bond_idx]))
                    cutoff_list.append(cutoff)
                    print(f"cutoff = {cutoff}")
                    self.canonicalize(cano_eps=1e-10, eigen_eps=1e-14, maxiter = 50, cano_scheme=cano_scheme)


            
    def build_rho_1s(self):
        rho_1s_rule = self.contraction_rule["rho_1s"]
        
        lam_square = [self.lam[i]**2 for i in range(self.z)]
        rho_A = oe.contract(rho_1s_rule, self.T1, self.T1.conj(), *lam_square)
        rho_B = oe.contract(rho_1s_rule, self.T2, self.T2.conj(), *lam_square)
        
        return rho_A, rho_B


    def parse_swap_gates(self, crossing_legs):
        swap_tensor_list = []
        for cross_i in crossing_legs:
            swap_i = torch.ones(len(self.parity_tabel[self.z]), len(self.parity_tabel[cross_i]), dtype=self.dtype, device=self.device)
            minus_parity_phy1 = np.sum(self.parity_tabel[self.z]==-1)
            minus_parity_cross_i = np.sum(self.parity_tabel[cross_i]==-1)
            swap_i[:minus_parity_phy1, :minus_parity_cross_i] = self.swap_on
            
            swap_tensor_list.append(swap_i)

        return swap_tensor_list
        
    def build_rho_2s_bondi(self, bond_idx):
        rho_2s_rule_i = self.contraction_rule["rho_2s"][f"{bond_idx}"]["einsum_str"]
        
        rho_2s_rule_i_crossing_1 = self.contraction_rule["rho_2s"][f"{bond_idx}"]["1_crossing"]
        rho_2s_rule_i_crossing_2 = self.contraction_rule["rho_2s"][f"{bond_idx}"]["2_crossing"]
        rho_2s_rule_i_crossing_3 = self.contraction_rule["rho_2s"][f"{bond_idx}"]["3_crossing"]
        rho_2s_rule_i_crossing_4 = self.contraction_rule["rho_2s"][f"{bond_idx}"]["4_crossing"]
        
        swap_tensor_list_1 = self.parse_swap_gates(rho_2s_rule_i_crossing_1)
        swap_tensor_list_2 = self.parse_swap_gates(rho_2s_rule_i_crossing_2)
        swap_tensor_list_3 = self.parse_swap_gates(rho_2s_rule_i_crossing_3)
        swap_tensor_list_4 = self.parse_swap_gates(rho_2s_rule_i_crossing_4)

        center_leg_tensor = None        
        outer_legs_tensor = []
        
        for i in range(self.z):
            if i == bond_idx:
                center_leg_tensor = self.lam[i]
            else:
                outer_legs_tensor.append(self.lam[i])
                
        
        rho_2s_i = oe.contract(rho_2s_rule_i, 
                               self.T1, *swap_tensor_list_1, *outer_legs_tensor,
                               center_leg_tensor,
                               self.T2, *swap_tensor_list_2, *outer_legs_tensor,
                               
                               self.T1.conj(), *swap_tensor_list_3, *outer_legs_tensor,
                               center_leg_tensor,
                               self.T2.conj(), *swap_tensor_list_4, *outer_legs_tensor)
        
        return rho_2s_i
    
    def build_rho_2s(self):
        return [self.build_rho_2s_bondi(i) for i in range(self.z)]
    
    def measure_bond_en(self):
        bond_en = []
        for idx,rho_2s in enumerate(self.build_rho_2s()):
            en = oe.contract("abcd, abcd ->", rho_2s, self.model_ham(self.coupling[idx]))
            bond_en.append(en)
        return bond_en
        
    def measure_occupy(self):
        rho_A, rho_B = self.build_rho_1s()
        
        n_up_A = torch.trace(rho_A @ self.n_up)
        n_down_A = torch.trace(rho_A @ self.n_down)
        
        n_up_B = torch.trace(rho_B @ self.n_up)
        n_down_B = torch.trace(rho_B @ self.n_down)
        
        sz_A = torch.trace(rho_A @ self.sz)
        sx_A = torch.trace(rho_A @ self.sx)

        sz_B = torch.trace(rho_B @ self.sz)        
        sx_B = torch.trace(rho_B @ self.sx)
        
        
        n_docc_A = torch.trace(rho_A @ self.n_double_occ)
        n_docc_B = torch.trace(rho_B @ self.n_double_occ)
        
        # print(f"n_up_A = {n_up_A}, n_up_B = {n_up_B}, n_down_A = {n_down_A}, n_down_B = {n_down_B}, n_docc_A = {n_docc_A}, n_docc_B = {n_docc_B}")
        print(f"sz_A = {sz_A}, sz_B = {sz_B}, sx_A = {sx_A}, sx_B = {sx_B}, n_up_A = {n_up_A}, n_up_B = {n_up_B}, n_down_A = {n_down_A}, n_down_B = {n_docc_B}, n_docc_A = {n_docc_A}, n_docc_B = {n_docc_B}")
        print(f"n_tol_A = {n_up_A + n_down_A}, n_tol_B = {n_up_B + n_down_B}")
        print(f"n_tol_avg = {(n_up_A + n_down_A + n_up_B + n_down_B)/2}")

        
        return sz_A, sx_A, n_up_A, n_down_A, n_docc_A,   sz_B, sx_B, n_up_B, n_down_B, n_docc_B
    
    
    def evolve_with_chemical_potential(self, mu, counting_normalize_factor=False):
        adjust_chemical_potential_op = torch.linalg.matrix_exp(self.beta*mu*self.n_tot/2)
        normalize_factor = torch.linalg.norm(adjust_chemical_potential_op)
        adjust_chemical_potential_op_normalized = adjust_chemical_potential_op / normalize_factor
        
        # NOTE: if we want to calculate the lnZ alway from half-filling, remember to count the normalize_factor
        if counting_normalize_factor:
            self.lnZ += torch.log(normalize_factor) * 2 # coeff 2 comes from double-layer structre, \rho(\beta/2) \rho(\beta/2)^\dagger.
        
        self.T1 = oe.contract("abc de, ef ->abc df", self.T1, adjust_chemical_potential_op_normalized)
        self.T2 = oe.contract("abc de, ef ->abc df", self.T2, adjust_chemical_potential_op_normalized)
        
        self.canonicalize(cano_eps=1e-10, eigen_eps=1e-14, maxiter = 50, cano_scheme="bp")
        
    def determine_chemical_potential(self, target_ntot, max_opt_iter=100):
        
        def loss_fn(mu):
            
            # save the current state to avoid the optimization process destroy the current state
            self.backup_push()
            
            self.evolve_with_chemical_potential(mu, counting_normalize_factor=False)

            rho_A, rho_B = self.build_rho_1s()
            
            n_A_adjusted = torch.trace(rho_A @ self.n_tot)
            n_B_adjusted = torch.trace(rho_B @ self.n_tot)
            
            n_avg_adjusted = (n_A_adjusted + n_B_adjusted)/2
            
            self.backup_pop()

            print(f"mu = {mu}, n_avg_adjusted = {n_avg_adjusted}, target_ntot = {target_ntot}")

            return n_avg_adjusted - target_ntot

        
        # 二分法试探最佳的mu
        mu_min = 0
        mu_max = np.sign(target_ntot-1) * self.mu_max
        eps=1e-9
                    
        if loss_fn(mu_min) * loss_fn(mu_max) > 0:
            print(f"mu_min = {mu_min}, mu_max = {mu_max}")
            print(f"loss_fn(mu_min) = {loss_fn(mu_min)}, loss_fn(mu_max) = {loss_fn(mu_max)}")
            # raise ValueError("初始区间必须包含一个零点（即 low 和 high 在函数值上异号）")
            return np.nan
        
        for i in range(max_opt_iter):
            mid = (mu_min + mu_max) / 2
            f_mid = loss_fn(mid)
            
            if abs(f_mid) < eps:
                mu_sol = mid
                break
            
            if loss_fn(mu_min) * f_mid < 0:
                mu_max = mid
            else:
                mu_min = mid

        mu_sol = (mu_min+mu_max)/2

        print(f"return sol = {mu_sol}, loss = {loss_fn(mu_sol):.4e}")
        
        return mu_sol
    
    
    def measure_entanglement_entropy(self):
        SE_list = []
        
        for i in range(self.z):
            lam_i = self.lam[i]
            SE_i  = -torch.vdot(lam_i**2 , torch.log(lam_i**2))
            SE_list.append(SE_i.cpu().item())
            
        return SE_list

    def evolve(self, filling, n, tau, ckp_dirname=None, ckp_prefix=None, ckp_save_interval=1, measure_step=5, obs_filename=None, cano_scheme="simple", trotter_scheme="1"):
        for i in range(n):
            self.evolve_once_v2(tau, cano_scheme=cano_scheme, trotter_scheme=trotter_scheme)

            self.beta += tau * 2
            self.evolve_step += 1
            
            if i % measure_step == 0:
                if np.abs(filling-1)>1e-7:
                    delta_mu = self.determine_chemical_potential(target_ntot=filling, max_opt_iter=200)
                    self.evolve_with_chemical_potential(delta_mu, counting_normalize_factor=True)
                    
                    self.summation_beta_mu += self.beta * delta_mu
                    self.chemical_potential = self.summation_beta_mu / self.beta

                    self.mu_max = np.abs(delta_mu) * 10
                
                free_energy = -1/self.beta * self.lnZ
                print(f"free_energy = {free_energy}")
                
                self.obs_list.append([self.evolve_step, 1/self.beta, free_energy.item(), self.chemical_potential])
                    
                if ckp_dirname is not None:
                    self.save_checkpoint(ckp_dirname, ckp_prefix, ckp_save_interval)
                
            
            
    def save_obs(self, filename):
        np.savetxt(filename, self.obs_list)
        print(f"Observation data saved to {filename}")
        
    def backup_push(self):
        self.obs_list_backup = self.obs_list.copy()
        self.T1_backup = self.T1.clone()
        self.T2_backup = self.T2.clone()
        
        
        self.lam_backup = deepcopy(self.lam)
        self.parity_tabel_backup = deepcopy(self.parity_tabel)
        self.beta_backup = deepcopy(self.beta)
        self.evolve_step_backup = deepcopy(self.evolve_step)
        self.lnZ_backup = deepcopy(self.lnZ)
        
    def backup_pop(self):
        self.obs_list = self.obs_list_backup
        self.T1 = self.T1_backup
        self.T2 = self.T2_backup
        self.lam = self.lam_backup
        self.parity_tabel = self.parity_tabel_backup
        self.beta = self.beta_backup
        self.evolve_step = self.evolve_step_backup
        self.lnZ = self.lnZ_backup
        
        self.obs_list_backup = None
        self.T1_backup = None
        self.T2_backup = None
        self.lam_backup = None
        self.parity_tabel_backup = None
        self.beta_backup = None
        self.evolve_step_backup = None
        self.lnZ_backup = None
        
    def save_checkpoint(self, dirname, prefix, save_interval):
        if self.evolve_step % save_interval != 0:
            return

        if not os.path.exists(dirname):
            os.makedirs(dirname)
        
        if self.calc_mode == "gs":
            save_path = f"{dirname}/{prefix}_gs.pt"
        elif self.calc_mode == "thermal":
            save_path = f"{dirname}/{prefix}_{self.evolve_step}.pt"
        
        torch.save(
            {
                "obs_list": self.obs_list[-1],
                "T1": self.T1,
                "T2": self.T2,
                "lam": self.lam,
                "parity_tabel": self.parity_tabel,
            }, save_path)
    
    def load_checkpoint(self, dirname, prefix, evolve_step):
        ckp_data = torch.load(f"{dirname}/{prefix}_{evolve_step}.pt", map_location=self.device, weights_only=False)
        self.T1 = ckp_data["T1"]
        self.T2 = ckp_data["T2"]
        self.lam = ckp_data["lam"]        
        self.parity_tabel = ckp_data["parity_tabel"]
        self.obs_list = [ckp_data["obs_list"]]
        
        self.evolve_step = self.obs_list[-1][0]
        self.beta        = 1 / self.obs_list[-1][1]
        self.lnZ         = -self.obs_list[-1][2]*self.beta
        
        print(f"Checkpoint loaded from {dirname}/{prefix}_{evolve_step}.pt")
        
        print(*self.obs_list)

    # def check_parity_preserving(self):
    #     for idx,px in enumerate(self.parity_tabel["x"]):
    #         for idy,py in enumerate(self.parity_tabel["y"]):
    #             for idz,pz in enumerate(self.parity_tabel["z"]):
    #                 for idphy1,pphy1 in enumerate(self.parity_tabel["phy"]):
    #                     for idphy2,pphy2 in enumerate(self.parity_tabel["phy"]):
    #                         if px*py*pz*pphy1*pphy2 == -1:
    #                             assert self.T1[idx,idy,idz,idphy1,idphy2] == 0, f"parity not preserved at ({idx},{idy},{idz},{idphy1},{idphy2})"
    #                             assert self.T2[idx,idy,idz,idphy1,idphy2] == 0, f"parity not preserved at ({idx},{idy},{idz},{idphy1},{idphy2})"
    #     print("[PARITY PERSERVING TEST] passed.")
    

    def verify_canonical_i(self, bond_idx):
        verify_canonical_rule = self.contraction_rule["verify_canonical_rule"].get(f"{bond_idx}", None)
        
        bond_tensor_square_list = []
        bond_tensor_square_complete_list = []
        for i in range(self.z):
            bond_tensor_square_complete_list.append(self.lam[i]**2)
            if i != bond_idx:
                bond_tensor_square_list.append(self.lam[i]**2)

        T1map_res_tensor = oe.contract(verify_canonical_rule, self.T1, self.T1,  *bond_tensor_square_list)
        T2map_res_tensor = oe.contract(verify_canonical_rule, self.T2, self.T2,  *bond_tensor_square_list)
        
        diff1 = torch.norm(T1map_res_tensor-torch.eye(T1map_res_tensor.shape[0], dtype=self.dtype, device=self.device))
        diff2 = torch.norm(T2map_res_tensor-torch.eye(T2map_res_tensor.shape[0], dtype=self.dtype, device=self.device))

        return diff1, diff2

    def verify_canonical(self, threshold=1e-9, verbose=True):
        diff1_list = []
        diff2_list = []
        for bond_idx in range(self.z):
            diff1, diff2 = self.verify_canonical_i(bond_idx)
            diff1_list.append(diff1)
            diff2_list.append(diff2)
        
        diff_max = max(max(diff1_list), max(diff2_list))
        flag     = (diff_max < threshold)
        if verbose:
            print(f"***** GEVP TEST [{flag}] *****")
            print(f"diff1_list = ", ", ".join([f"{diff:.4e}" for diff in diff1_list]))
            print(f"diff2_list = ", ", ".join([f"{diff:.4e}" for diff in diff2_list]))
            print("***** **************** *****")
        
        return flag


    
    def boundary_matvec_i(self, alpha, bond_idx, trial_vec):
        """
        bond_idx: the outer bond index
        """
        gevp_matvec_rule = self.contraction_rule["gevp_matvec_rule"].get(f"{bond_idx}", None)
        
        bond_tensor_list = []
        trial_bond_tensor_list = []
        for i in range(self.z):
            if i != bond_idx:
                bond_tensor_list.append(self.lam[i])
                trial_bond_tensor_list.append(trial_vec[i])
        
        if alpha == 0: # T1
            res = torch.einsum(gevp_matvec_rule, self.T1, self.T1,  *bond_tensor_list, *bond_tensor_list, *trial_bond_tensor_list)
        elif alpha == 1: # T2
            res = torch.einsum(gevp_matvec_rule, self.T2, self.T2,  *bond_tensor_list, *bond_tensor_list, *trial_bond_tensor_list)
        else:
            raise ValueError("alpha should be 0 (T1) or 2 (T2)")
        
        # return matix-vector product (without renormalization)
        return res
    
    def boundary_matvec(self, initial_trial_vec=None):
        if initial_trial_vec is None:
            initial_trial_vec = []
            for alpha in range(2):
                for i in range(self.z):
                    # randn_gauge = torch.randn(self.lam[i].shape[0], self.lam[i].shape[0], dtype=self.dtype, device=self.device)
                    randn_gauge = torch.eye(self.lam[i].shape[0], dtype=self.dtype, device=self.device)
                    # message_tensor = randn_gauge@randn_gauge.T
                    message_tensor = randn_gauge
                    initial_trial_vec.append(message_tensor)

        res = []

        # mapping one sublatice to another
        trial_vec_alpha_list = [
            initial_trial_vec[self.z:2*self.z], # alpha = 0, calculating sublattice 1, use sublattice 2 as trial vector
            initial_trial_vec[0:self.z]  # alpha = 1, calculating sublattice 2, use sublattice 1 as trial vector
        ]
        
        for alpha, trial_vec_alpha in enumerate(trial_vec_alpha_list):
            for i in range(self.z):
                vi = self.boundary_matvec_i(alpha, i, trial_vec_alpha)
                vi /= torch.linalg.norm(vi)
                res.append(vi)
        
        return res

    def solve_boundary(self, eigen_eps=1e-12, maxiter=1000):
        boundary_tensor = None
        for idx in range(maxiter):
            boundary_tensor_new = self.boundary_matvec(boundary_tensor)

            if idx % 5 == 0 and idx > 0:
                diff = 0
                for i in range(2*self.z):
                    diff += torch.norm(boundary_tensor_new[i] - boundary_tensor[i])
                if diff < eigen_eps:
                    boundary_tensor = boundary_tensor_new
                    # print(f"Boundary tensor converged at {idx}th iteration.")
                    break
            boundary_tensor = boundary_tensor_new
            
        return boundary_tensor
    
    def fix_to_canonical_gauge(self, matvec_maxiter=1000, eigen_eps=1e-12):
        boundary_tensor = self.solve_boundary(eigen_eps=eigen_eps, maxiter=matvec_maxiter)

        u_alpha_i    = {}
        uinv_alpha_i = {}
        
        for alpha in range(2):
            for i in range(self.z):
                boundary_tensor_i = boundary_tensor[alpha*self.z + i]

                w,v = parity_preserving_eigh(boundary_tensor_i, self.parity_tabel[i])
                
                u     = v @ torch.diag(torch.sqrt(w))
                u_inv = torch.diag(safe_inv(torch.sqrt(w))) @ v.t()
                
                u_alpha_i[(alpha,i)]    = u
                uinv_alpha_i[(alpha,i)] = u_inv


        # FIX EIGENVECTOR TO IDENTITY
        T1_factor = []
        T2_factor = []
        for i in range(self.z):
            combine_i = oe.contract("ba,b,bc->ac", u_alpha_i[(0,i)], self.lam[i], u_alpha_i[(1,i)])
            
            U,S,VH = parity_preserving_svd_no_truncation(combine_i, self.parity_tabel[i])
            
            Snorm = torch.linalg.norm(S)
            self.lam[i] = S/Snorm
            self.lnZ += torch.log(Snorm)
            
            T1_factor.append(oe.contract("ac,ab -> bc", uinv_alpha_i[(0,i)], U))
            T2_factor.append(VH @ uinv_alpha_i[(1,i)])
            
        
        gauge_fixing_einsum = self.contraction_rule["canonical_gague_fixing_rule"]
        
        self.T1 = oe.contract(gauge_fixing_einsum, self.T1, *T1_factor)
        self.T2 = oe.contract(gauge_fixing_einsum, self.T2, *T2_factor)
        

        gevp_eigenvalue_rule = self.contraction_rule["gevp_eigenvalue_rule"]
        
        bond_tensor_square = [self.lam[i]**2 for i in range(self.z)]
        
        lam1 = oe.contract(gevp_eigenvalue_rule, self.T1, self.T1, *bond_tensor_square)
        lam2 = oe.contract(gevp_eigenvalue_rule, self.T2, self.T2, *bond_tensor_square)
        
        
        self.T1 /= torch.sqrt(lam1)
        self.T2 /= torch.sqrt(lam2)
        
        self.lnZ += (torch.log(lam1) + torch.log(lam2))/2
        

    
    # def measure_correlation_length(self):
        
    #     Identity_list = [torch.eye(self.T1.shape[i], dtype=self.dtype, device=self.device) for i in range(self.z)]
        
        
    #     def __transfer_0_1__(transfer_tensor_initial: np.array) -> np.array:
    #         transfer_tensor_initial_torch = torch.from_numpy(transfer_tensor_initial).to(self.device)
    #         transfer_tensor_initial_torch = torch.reshape(transfer_tensor_initial_torch, (Identity_list[0].shape[0], Identity_list[0].shape[0]))       
    #         # a: 0 --> 1
    #         transfer_tensor_mid_torch   = self.boundary_matvec_i(0, 1, [transfer_tensor_initial_torch, None]+Identity_list[2:])
    #         # b: 1 --> 0
    #         transfer_tensor_final_torch = self.boundary_matvec_i(1, 0, [None, transfer_tensor_mid_torch]+Identity_list[2:])

    #         transfer_tensor_final = transfer_tensor_final_torch.cpu().numpy()

    #         return transfer_tensor_final
        
    #     op = LinearOperator((Identity_list[0].shape[0]**2, Identity_list[0].shape[0]**2), matvec=__transfer_0_1__)
        
    #     w  = eigs(op, k=2, which="LM", return_eigenvectors=False)
    #     sorted_inds = np.argsort(np.abs(w))[::-1]
    #     w  = w[sorted_inds]
        
    #     xi = 2 / np.log(abs(w[0])/abs(w[1]))
    #     print(f"Correlation Length = {xi}   |  lambda_0 = {w[0]}, lambda_1 = {w[1]}")
        
    #     return xi      

    # def measure_corr_2p(self, path_list:list, op, start_type=0):
    #     assert self.z == 3, "Only support z=3 now."
    #     """
    #     Measure the two-point correlation along a specific path.

    #     Args:
    #         path_list (list): A Python list containing the path information. For example, 
    #             to measure correlation between points pi and pj along a specific path, 
    #             use a list like [0, 1, 2, 1], which indicates:
    #             - p0 --xbond--> p1 
    #             - p1 --ybond--> p2 
    #             - p2 --zbond--> p3 
    #             - p3 --ybond--> p4

    #         op (torch.Tensor): The operator to be measured.
            
    #         start_type (str): The starting tensor type. It can be either 0 or 1. (sublattice A or B)
    #     Notes:
    #         - Ensure the canonicalization process is completed before calling this function.
    #     """

    #     alpha_list = [ (start_type+i)%2 for i in range(len(path_list) + 1)]
        
    #     eye_x = torch.eye(self.T1.shape[0], dtype=self.dtype, device=self.device)
    #     eye_y = torch.eye(self.T1.shape[1], dtype=self.dtype, device=self.device)
    #     eye_z = torch.eye(self.T1.shape[2], dtype=self.dtype, device=self.device)
        
    #     trial_vec = [eye_x, eye_y, eye_z]
        
    #     for alpha_i in range(len(alpha_list)):
    
    #         if alpha_i == len(alpha_list) - 1:
    #             out_bond_idx = (path_list[alpha_i-1] + 1) % self.z
    #         else:
    #             out_bond_idx = path_list[alpha_i]

    #         matvec_rule = self.contraction_rule["gevp_matvec_rule"].get(f"{out_bond_idx}") 
                
    #         bond_tensor_list = []
    #         trial_bond_tensor_list = []
                
    #         for bond_i in range(self.z):
    #             if bond_i != out_bond_idx:
    #                 bond_tensor_list.append(self.lam[bond_i])
    #                 trial_bond_tensor_list.append(trial_vec[bond_i])
            
    #         alpha = alpha_list[alpha_i]    
    #         if alpha == 0: # T1
            
    #             if alpha_i == 0 or alpha_i == len(alpha_list) - 1:
    #                 T1_dressed = oe.contract("abc de, df-> abc fe", self.T1, op) # slow, can be optimized
    #             else:
    #                 T1_dressed = self.T1
            
    #             res = torch.einsum(matvec_rule, T1_dressed, self.T1,  *bond_tensor_list, *bond_tensor_list, *trial_bond_tensor_list)
            
    #         elif alpha == 1: # T2
            
    #             if alpha_i == 0 or alpha_i == len(alpha_list) - 1:
    #                 T2_dressed = oe.contract("abc de, df-> abc fe", self.T2, op) # slow, can be optimized
    #             else:
    #                 T2_dressed = self.T2
            
    #             res = torch.einsum(matvec_rule, T2_dressed, self.T2,  *bond_tensor_list, *bond_tensor_list, *trial_bond_tensor_list)
            
    #         # update trial vector
    #         trial_vec = [eye_x, eye_y, eye_z]
    #         # if alpha_i == 0:
    #         #     trial_vec[out_bond_idx] = res * 3
    #         # else:
    #         #     trial_vec[out_bond_idx] = res * 2
    #         trial_vec[out_bond_idx] = res
                     
    #     corr_val = oe.contract("aa,a->", res, self.lam[0]**2)
        
    #     return corr_val
    