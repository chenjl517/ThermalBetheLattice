from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


from argparse import ArgumentParser
# from BetheLattice import Bethelattice
import BetheLattice
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, exists


parser = ArgumentParser()
parser.add_argument('-t', type=float, default=1)
parser.add_argument('-J', type=float, default=0)
parser.add_argument('-U', type=float, default=0)
parser.add_argument('-z', type=int, default=3)
parser.add_argument('-chi', type=int, default=6)
parser.add_argument('-Ntau', type=int, default=100)
parser.add_argument('-tau', type=float, default=0.05)
parser.add_argument('-dmu', type=float, default=0)
parser.add_argument('-dh', type=float, default=0)
parser.add_argument('-dhs', type=float, default=0)
parser.add_argument('-re_measure', type=bool, default=True)

parser.add_argument('-prefix', type=str, default="")
parser.add_argument('-filling', type=float, default=1)


# parser.add_argument('-evolveIndexStart', type=int, default=1)
# parser.add_argument('-evolveIndexEnd', type=int, default=100)
# parser.add_argument('-evolveIndexStep', type=int, default=2)

def set_num_thread(n):
    import os
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
    os.environ["NUMBA_NUM_THREADS"] = str(n)
    torch.set_num_threads(n)

if __name__ == "__main__":
    import torch

    # from hubbard_ham import build_ham, build_gate
    from phy_model.tJU_ham import build_ham, build_gate
    
    set_num_thread(4)
    
    args = parser.parse_args()
    print(args)
    
    device = torch.device("cpu")
    
    calc_mode = "thermal"
    
    # Hubbard
    ham          = lambda t_dir: build_ham(t_dir, J=args.J, U = args.U, mu = 0, h = args.dh, hs = args.dhs, z = args.z, dtype=torch.float64, device=device)
    gate         = lambda tau, t_dir: build_gate(tau, t_dir, J=args.J, U = args.U, mu = args.U/2 + args.dmu, h = args.dh, hs = args.dhs, z = args.z, dtype=torch.float64, device=device)
    basis_parity = [-1, -1, 1, 1]

    tx, ty, tz = args.t, args.t, args.t
    
    bethe_lattice_hubbard = BetheLattice.BetheLattice(z=args.z, coupling_x=tx, coupling_y=ty, coupling_z=tz, gate=gate, ham=ham, ham_basis_parity=basis_parity, chemical_potential=args.U/2, chi_max=args.chi, ignore_threshold=1e-7, statistic="Fermion", calc_mode=calc_mode, dtype=torch.float64, device=device)
    
    if abs(args.dmu) < 1e-15 and abs(args.dh) < 1e-15 and abs(args.dhs) < 1e-15: # dmu = 0
        model_info  = f"{calc_mode}_z={args.z}_t={args.t:.4f}_J={args.J:.4f}_U={args.U:.4f}_tau={args.tau:.4e}_chi={args.chi}_filling={args.filling}"
    else:
        model_info  = f"{calc_mode}_z={args.z}_t={args.t:.4f}_J={args.J:.4f}_U={args.U:.4f}_tau={args.tau:.4e}_chi={args.chi}_dmu={args.dmu:.4e}_dh={args.dh:.4e}_dhs={args.dhs:.4e}"


    ckp_dirname = f"ckp_{args.prefix}/{model_info}"
    ckp_prefix = f"{model_info}_checkpoint"
    
    data_obs = []
    
    print(f"MPI INFO: rank {rank} of {size}")
    if rank == 0:
        files_idx = [f.split("_")[-1].split(".")[0] for f in listdir(ckp_dirname) if isfile(join(ckp_dirname, f))]
        sorted_files_idx = sorted(files_idx, key=lambda x: int(x))
        spilted_files_idx = np.array_split(sorted_files_idx, size)
    else:
        spilted_files_idx = None
    
    local_tasks = comm.scatter(spilted_files_idx, root=0)        
    print(f"rank {rank} has tasks: {local_tasks}")
    
    def measure_task(evolveIndex):
        bethe_lattice_hubbard.load_checkpoint(ckp_dirname, ckp_prefix, evolveIndex)
        obs_item = []
        for item in bethe_lattice_hubbard.obs_list[-1]:
            if type(item) == torch.Tensor:
                obs_item.append(item.cpu().item())
            else:
                obs_item.append(item)
    
        if args.re_measure:
            # if len(obs_item) != 4:
            #     print("(#. T ,FN) not found !")
            #     obs_item = [np.nan, np.nan, np.nan, np.nan]
            # else:
            #     obs_item = [obs_item[0], obs_item[1], obs_item[2], obs_item[3]] 
            # obs_item = [obs_item[0], obs_item[1], obs_item[2], obs_item[3]] # No., T, FN, mu, 
            obs_item = [obs_item[3], obs_item[1], obs_item[2]] # mu, T, FN 
            
            # xi = bethe_lattice_hubbard.calc_correlation_length()
            # obs_item.append(xi)
            
            bondx, bondy, bondz = bethe_lattice_hubbard.measure_bond_en() 
            # obs_item[3:6] = bondx.cpu().item(), bondy.cpu().item(), bondz.cpu().item()
            obs_item += [bondx.cpu().item(), bondy.cpu().item(), bondz.cpu().item()] # 4,5,6
        
            # n_up_1, n_down_1, n_docc_1, n_up_2, n_down_2, n_docc_2 = bethe_lattice_hubbard.measure_occupy()
            sz_1, sx_1, n_up_1, n_down_1, n_docc_1,   sz_2, sx_2, n_up_2, n_down_2, n_docc_2 = bethe_lattice_hubbard.measure_occupy()
            obs_item += [sz_1.cpu().item(), sx_1.cpu().item(), n_up_1.cpu().item(), n_down_1.cpu().item(), n_docc_1.cpu().item()] # 7,8,9,10,11
            obs_item += [sz_2.cpu().item(), sx_2.cpu().item(), n_up_2.cpu().item(), n_down_2.cpu().item(), n_docc_2.cpu().item()] # 12,13,14,15
            
            #  0, 1, 2,   3,   4,    5,    6,      7,        8,        9,       10,   11,     12,        13,       14,       15
            # mu, T, F, E_1, E_2,  E_3, sz_1,   sx_1,   n_up_1, n_down_1, n_docc_1, sz_2,   sx_2,    n_up_2, n_down_2, n_docc_2
            
            # obs_item[6:12] = n_up_1.cpu().item(), n_down_1.cpu().item(), n_docc_1.cpu().item(), n_up_2.cpu().item(), n_down_2.cpu().item(), n_docc_2.cpu().item()
            # obs_item[6:12] = sz_1.cpu().item(), sx_1.cpu().item(), n_docc_1.cpu().item(), sz_2.cpu().item(), sx_2.cpu().item(), n_docc_2.cpu().item()

            # measure entanglement entropy
            SE_x, SE_y, SE_z = bethe_lattice_hubbard.measure_entanglement_entropy()
            obs_item += [SE_x, SE_y, SE_z]

            
            # measure correlation length
            # xi = bethe_lattice_hubbard.measure_correlation_length()
            # obs_item.append(xi)

        return obs_item

    local_results = [measure_task(evolveIndex) for evolveIndex in local_tasks]
    
    results_gathered = comm.gather(local_results, root=0)
    
    if rank == 0:
        data_obs = []
        for result in results_gathered:
            data_obs += result
        
        save_dir = f"data/{args.prefix}"
        path     = f"{save_dir}/{model_info}_obs_mpisize={size}.txt"
        # check if the directory exists
        if not exists(save_dir):
            makedirs(save_dir)
            
        np.savetxt(f"{save_dir}/{model_info}_obs_mpisize={size}.txt", data_obs)

"""
obs format:
   0, 1, 2,   3,   4,   5,    6,    7,      8,        9,       10,   11,   12,     13,       14,       15    16,   17,   18
step, T, F, E_1, E_2, E_3, sz_1, sx_1, n_up_1, n_down_1, n_docc_1, sz_2, sx_2, n_up_2, n_down_2, n_docc_2, SE_x, SE_y, SE_z
"""

