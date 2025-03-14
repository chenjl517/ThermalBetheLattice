import torch
import numpy as np
from textwrap import dedent
# import matplotlib.pyplot as plt
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

def stable_svd(mat, eps = 1e-6):
    try:
        u, s, vh = torch.linalg.svd(mat)
        return u, s, vh
    except Exception as e:
        print(f"Error: {e}\nTry to add eps = {eps} to the matrix")
        u, s, vh = torch.linalg.svd(mat + eps)
        return u, s, vh

# @profile
def parity_preserving_svd(mat, parity_row:np.array, parity_col:np.array, chi_max, singular_threshold, force_keep=False, verbose=False, dask_client=None):
    # assert mat.shape[0] == mat.shape[1]
    
    # #===============================
    # normal_svd
    # u_normal, s_normal, vh_normal = torch.linalg.svd(mat)
    # s_skiiped = s_normal[s_normal > singular_thershold]
    # chi_max = min(chi_max, len(s_skiiped))
    
    # return u_normal[:,:chi_max], s_skiiped[:chi_max], vh_normal[:chi_max,:], np.array([1]*chi_max,dtype=np.int8), torch.sum(s_skiiped[chi_max:]**2)
    # # #===============================
    
    # print(f"s_normal = {s_normal}")
    # input()
    # if s_normal.max() > 1:
    #     print(mat.abs().max())
    # client = Client(process=False)
    
    # print(f"svd-mat.shape = {mat.shape}")
    
    odd_parity_num_row = np.sum(parity_row<0)
    odd_parity_num_col = np.sum(parity_col<0)
    
    
    sorted_p_row_inds = np.argsort(parity_row)
    sorted_p_col_inds = np.argsort(parity_col)
    
    sorted_p_row_inv = calc_order_inv(sorted_p_row_inds)
    sorted_p_col_inv = calc_order_inv(sorted_p_col_inds)
    
    # assert odd_parity_num_row >= 1
    # assert odd_parity_num_col >= 1
    # assert chi_max >= 2
    
    mat = mat[sorted_p_row_inds,:]
    mat = mat[:,sorted_p_col_inds]
    
    # try:
    #     assert check_block_diag_form(mat, odd_parity_num_row, odd_parity_num_col), "The input matrix is not in block diagonal form"
    # except Exception as e:
    #     print(parity_row)
    #     print(parity_col)
    #     print(mat.abs().max())
    #     mat10 = mat[odd_parity_num_row:, :odd_parity_num_col]
    #     mat01 = mat[:odd_parity_num_row, odd_parity_num_col:]
    #     # get the max value pos
    #     pos10 = torch.argmax(mat10.abs())
    #     pos01 = torch.argmax(mat01.abs())
    #     print("row = ", odd_parity_num_row, "col = ", odd_parity_num_col)
    #     print(mat10.abs().max(), pos10)
    #     print(mat01.abs().max(), pos01)
    #     raise e
        
    mat_p_odd = mat[:odd_parity_num_row, :odd_parity_num_col]
    mat_p_even = mat[odd_parity_num_row:, odd_parity_num_col:]
    
    if dask_client is None:
        u_odd, s_odd, vh_odd    = stable_svd(mat_p_odd)
        u_even, s_even, vh_even = stable_svd(mat_p_even)
    else:
        parallel_svd = dask_client.map(stable_svd, [mat_p_odd, mat_p_even])
        parallel_svd_res = dask_client.gather(parallel_svd)

        u_odd, s_odd, vh_odd    = parallel_svd_res[0]
        u_even, s_even, vh_even = parallel_svd_res[1]
        

    s_odd_skiped = s_odd[s_odd > singular_threshold]
    s_even_skiped = s_even[s_even > singular_threshold]
    
    chi_cut = min(chi_max, len(s_odd_skiped)+len(s_even_skiped))
    
    s_combined = torch.cat([s_odd_skiped, s_even_skiped])
    p_combined = [-p_odd for p_odd in range(1,1+len(s_odd_skiped))] + [p_even for p_even in range(1,1+len(s_even_skiped))]
    
    
    sorted_s_combined_order = torch.argsort(s_combined, descending=True)
    # print(f" sorted s combined order = {sorted_s_combined_order}")
    
    p_combined_sorted = [p_combined[i] for i in sorted_s_combined_order]
    
    # avoid cutting at the edge
    if chi_cut < len(s_combined):
        chi_cut0 = chi_cut
        rdiff = lambda a,a0: torch.abs(a-a0)/torch.abs(a0)
        while( rdiff(s_combined[chi_cut-1], s_combined[chi_cut]) < 1e-7 ):
            # chi_cut += 1
            chi_cut -= 1
            # print(f"increasing chi_cut to {chi_cut}")
            if chi_cut == 0:
                # reach the boundary 0, return the original chi_cut
                chi_cut = chi_cut0 
                # print(f"reach the boundary, chi_cut = {chi_cut}")
                break

    p_combined_cuted  = p_combined_sorted[:chi_cut]
    
    odd_parity_num = len([p for p in p_combined_cuted if p < 0])
    even_parity_num = len([p for p in p_combined_cuted if p >0])


    #=================================================

    if force_keep:    
        if odd_parity_num == 0:
            odd_parity_num = 1
            # even_parity_num -= 1
        if even_parity_num == 0:
            even_parity_num = 1
            # odd_parity_num -= 1


    u_combined = torch.zeros(mat.shape[0], chi_cut)
    vh_combined = torch.zeros(chi_cut, mat.shape[1])
    
    if verbose:
        print(dedent(f"""\
            
            ===================================================
              odd_parity_compelete = {s_odd},
              even_parity_compelete = {s_even},
              ---
                  chi_cut = {chi_cut},
                  odd_parity_singular_values = {s_odd_skiped[:odd_parity_num]},
                  even_parity_singular_values = {s_even_skiped[:even_parity_num]},
                  odd_parity_num = {odd_parity_num}, even_parity_num = {even_parity_num}
            ===================================================

              """))
    
    u_combined = torch.block_diag(u_odd[:,:odd_parity_num], u_even[:,:even_parity_num])
    vh_combined = torch.block_diag(vh_odd[:odd_parity_num,:], vh_even[:even_parity_num,:])
    
    s_combined = torch.cat([s_odd_skiped[:odd_parity_num], s_even_skiped[:even_parity_num]])
    # s_combined = torch.cat([s_odd[:odd_parity_num], s_even[:even_parity_num]])
    
    cut_parity = np.array([-1]*odd_parity_num + [1]*even_parity_num, dtype=parity_row.dtype)
    
    # plt.figure()
    # plt.subplot(121)
    # plt.matshow(u_combined)
    # plt.subplot(122)
    # plt.matshow(vh_combined)
    # plt.savefig("u_vh_combined.png")
    
    # adjust the order of u_combined, vh_combined back
    u_combined = u_combined[sorted_p_row_inv,:]
    vh_combined = vh_combined[:,sorted_p_col_inv]
    
    # recovered_row_parity = parity_row[sorted_p_row_inds][sorted_p_row_inv]
    # recovered_col_parity = parity_col[sorted_p_col_inds][sorted_p_col_inv]
    
    # print(all(recovered_row_parity == parity_row), all(recovered_col_parity == parity_col))
    # print(f"original row parity  = \n{parity_row}")    
    # print(f"recovered_row_parity = \n{recovered_row_parity}")
    # print(f"original col parity  = \n{parity_col}")
    # print(f"recovered_col_parity = \n{recovered_col_parity}")
    
    # DEBUG: check u_combined parity preserving
    # for x,px in enumerate(parity_row):
    #     for y, py in enumerate(cut_parity):
    #         if px*py == -1:
    #             assert u_combined[x,y] == 0, f"[after permute], u_combined[{x},{y}] = {u_combined[x,y]}"
    
    # for x,px in enumerate(cut_parity):
    #     for y, py in enumerate(parity_col):
    #         if px*py == -1:
    #             assert vh_combined[x,y] == 0, f"[after permute],vh_combined[{x},{y}] = {vh_combined[x,y]}"
    
    
    
    trunc_err = torch.sum(s_odd[odd_parity_num:]**2) + torch.sum(s_even[even_parity_num:]**2)
    
    # print(f"s_combined = {s_combined}")
    # input()
    
    #============================== just for debug ==========================
    # import matplotlib.pyplot as plt
    # sorted_s_combined = torch.sort(s_combined, descending=True).values
    # plt.figure()
    # plt.plot(sorted_s_combined, 'o-')
    # plt.plot(s_skiiped, '--')
    # plt.savefig("may_be_ill_s.png", dpi=300)
    # print("\n\n****************************")
    # print(f"s_normal = {s_normal[:chi_max]}")
    # print(f"sorted(s_combined) = {sorted_s_combined}")
    # print("****************************\n\n")
    
    # just for debug =================================
    # if odd_parity_num != even_parity_num:
    # print(f"chi_cut = {chi_cut}, odd_parity_num = {odd_parity_num}, even_parity_num = {even_parity_num}")
    # print(f"cut_parity = {cut_parity}")
        
    return u_combined, s_combined, vh_combined, cut_parity, trunc_err
    
    
if __name__ == "__main__":
    from ParityTensor import ParityTensor
    pmat = ParityTensor(torch.rand(8,8, dtype=torch.float64, device='cpu'), parity_dict={0:[-1, 1, 1,-1, 1, -1, 1, 1], 1:[-1, 1, 1,-1, 1, -1, 1, 1]})
    pmat._delete_parity_destroying_elements()
    
    print(pmat.parity_dict)
    for cut in range(2, pmat.shape[0]+1):
        u,s,vh,odd_parity_num = parity_preserving_svd(pmat, pmat.parity_dict[0].numpy(), pmat.parity_dict[1].numpy(), chi_max=cut, singular_threshold=1e-14, verbose=False)
        
        mat_reconstructed = u @ torch.diag(s) @ vh
        diff = torch.linalg.norm(mat_reconstructed - pmat)
        
        print(f"cut = {cut}, diff = {diff}")
        
        print("--------------------------------------")