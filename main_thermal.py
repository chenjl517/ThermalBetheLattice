from argparse import ArgumentParser
from os import listdir
from os.path import isfile, join, exists
import BetheLattice
from textwrap import dedent


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
parser.add_argument('-filling', type=float, default=1)
parser.add_argument('-load', type=bool, default=False)

parser.add_argument('-line_mode', type=bool, default=False)
parser.add_argument('-lm_Tstart', type=float, default=1)
parser.add_argument('-lm_Tend', type=float, default=0.1)
parser.add_argument('-lm_Nstep', type=int, default=10)

parser.add_argument('-prefix', type=str, default="")
parser.add_argument('-cuda', type=int, default=-1)

def set_num_thread(n):
    import os
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
    os.environ["NUMBA_NUM_THREADS"] = str(n)
    torch.set_num_threads(n)

def calc_evolve_schedule(args):
    assert args.lm_Tstart > args.lm_Tend, "Tstart should be larger than Tend (cooling down)"
    # assert args.load == False, "Loading checkpoint is not supported in line_mod mode yet!"
    
    print(f"args.line_mode: {args.line_mode}")
    if args.line_mode:
        print(dedent("""
                    -------- [Warning] line_mod is activated; Ntau will be ignored! --------
                        The program will first evolve to T_i, the closest temperature satisfying T_i >= Tstart, using fixed steps of `args.tau`.  
                        It will then cool down to T_end in a linear schedule, with evolution steps determined automatically.
                        NOTE: A large `tau` in the schedule may introduce significant Trotter error. Ensure it is small enough to maintain accuracy.
                    ------------------------------------------------------------------------
                    """))
    else:
        return [args.tau] * args.Ntau
    
    # 2 comes from \rho(beta/2) * \rho^\dagger(beta/2), the symmetric cooling down
    N_fix_tau_end  = int(1 / args.lm_Tstart / args.tau / 2)
    T_fix_tau_end= 1 / (N_fix_tau_end * args.tau * 2)
    tau_transition = (1/args.lm_Tstart - 1/T_fix_tau_end) / 2
    
    T_interval_length = args.lm_Tstart - args.lm_Tend
    dT                = T_interval_length / args.lm_Nstep
    T_target          = [args.lm_Tstart - dT*i for i in range(1, args.lm_Nstep+1)]
    beta_target       = [1/T for T in T_target]
    
    tau_schedule = [args.tau] * N_fix_tau_end + [tau_transition]*(tau_transition > 0)
    beta0 = 1/args.lm_Tstart
    for beta_i in beta_target:
        tau_i = (beta_i - beta0)/2
        tau_schedule.append(tau_i)
        beta0 = beta_i
        print(f"beta_i: {beta_i}, tau_i: {tau_i}")
        
    print(f"tau_schedule: {tau_schedule}\n  --- min: {min(tau_schedule)}\n --- max: {max(tau_schedule)}")
    if max(tau_schedule) > args.tau:
        print("[Warning] The largest tau in the schedule is larger than the initial tau. Please be cautious with the Trotter error.")
    input("Press Enter to continue...")
    return tau_schedule
    
    
    
#================================================================================================
    


if __name__ == "__main__":
    import torch
    from phy_model.tJU_ham import build_ham, build_gate
    
    set_num_thread(8)
    
    args = parser.parse_args()
    print(args)
    
    device = torch.device("cuda:{}".format(args.cuda) if args.cuda >= 0 else "cpu")
    
    calc_mode = "thermal"
    
    # Hubbard
    ham          = lambda t_dir : build_ham(t_dir, J=args.J, U = args.U, mu = 0, h = args.dh, hs = args.dhs, z = args.z, dtype=torch.float64, device=device)
    gate         = lambda tau, t_dir : build_gate(tau, t_dir, J=args.J, U = args.U, mu = args.U/2 + args.dmu, h = args.dh, hs = args.dhs, z = args.z, dtype=torch.float64, device=device)
    basis_parity = [-1, -1, 1, 1]
    
    
    tx, ty, tz = args.t, args.t, args.t
    
    bethe_lattice_hubbard = BetheLattice.BetheLattice(z=args.z, coupling_x=tx, coupling_y=ty, coupling_z=tz, gate=gate, ham=ham, ham_basis_parity=basis_parity, chemical_potential=args.U/2, chi_max=args.chi, ignore_threshold = 1e-9, statistic="Fermion", calc_mode=calc_mode, dtype=torch.float64, device=device, measure=False)
    

    if abs(args.dmu) < 1e-15 and abs(args.dh) < 1e-15 and abs(args.dhs) < 1e-15: # dmu = 0
        model_info  = f"{calc_mode}_z={args.z}_t={args.t:.4f}_J={args.J:.4f}_U={args.U:.4f}_tau={args.tau:.4e}_chi={args.chi}_filling={args.filling}"
    else:
        model_info  = f"{calc_mode}_z={args.z}_t={args.t:.4f}_J={args.J:.4f}_U={args.U:.4f}_tau={args.tau:.4e}_chi={args.chi}_dmu={args.dmu:.4e}_dh={args.dh:.4e}_dhs={args.dhs:.4e}"
        
    
    ckp_dirname = f"ckp_{args.prefix}/{model_info}"
    ckp_prefix = f"{model_info}_checkpoint"
        
    if args.load:
        # file listing
        if exists(ckp_dirname):
            files = [f for f in listdir(ckp_dirname) if isfile(join(ckp_dirname, f))]

            if len(files) == 0:
                print("No checkpont founded! start from infinite high temperature!")
            else:
                last_ckp = sorted(files, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
                print(f"\nlast checkpoint: {last_ckp}\n")
                last_ckp_idx = int(last_ckp.split("_")[-1].split(".")[0])
                bethe_lattice_hubbard.load_checkpoint(ckp_dirname, ckp_prefix, last_ckp_idx)
        else:
            print("No checkpont founded! start from infinite high temperature!")


    tau_schedule = calc_evolve_schedule(args)
    
    for idx, tau in enumerate(tau_schedule):
        
        data_dir = args.prefix
        cano_scheme = "bp"

        print(f"evolve step: {idx}, canonicalize scheme: {cano_scheme}")
        bethe_lattice_hubbard.evolve(filling=args.filling, n = 1, tau = tau, ckp_dirname=ckp_dirname, ckp_prefix=ckp_prefix, measure_step=1,
                                 obs_filename=f"{data_dir}/{calc_mode}_z={args.z}_t={args.t:.4f}_J={args.J:.4f}_U={args.U:.4f}_Filling={args.filling:.4f}_tau={args.tau:.4e}_chi={args.chi}.txt", 
                                 cano_scheme=cano_scheme, trotter_scheme="1")
