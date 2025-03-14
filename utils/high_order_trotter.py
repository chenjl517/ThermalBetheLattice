# Optimised Trotter Decompositions for Classical and Quantum Computin
# https://arxiv.org/pdf/2211.02691.pdf

def get_scheme_coeff(scheme_name):
    
    if scheme_name == "2-2":
        print("Scheme 2-2: Omelyan 2")
        ci_list, di_list = calc_decomposition_coeff(*omelyan_order2())
    elif scheme_name == "4-1":
        print("Scheme 4-1: Suzuki-Trotter 4")
        ci_list, di_list = calc_decomposition_coeff(*suzuki_order4())
    elif scheme_name == "4-2":
        print("Scheme 4-2: Blanes-Moan 4")
        ci_list, di_list = calc_decomposition_coeff(*blanes_moan_order4())
        
    return ci_list, di_list
        

def calc_decomposition_coeff(ai_list:list, bi_list:list):
    length = len(ai_list)
    
    ai_table = dict(enumerate(ai_list))
    bi_table = dict(enumerate(bi_list))
    
    c0 = ai_table.get(0,0)
    d0 = bi_table.get(0,0) - c0
    
    ci_list = [c0]
    di_list = [d0]
    
    for i in range(1, length):
        ci = ai_table.get(i,0) - di_list[i-1]
        di = bi_table.get(i,0) - ci
        
        ci_list.append(ci)
        di_list.append(di)
    
    # drop zero elements
    ci_list = [x for x in ci_list if abs(x) > 1e-15]
    di_list = [x for x in di_list if abs(x) > 1e-15]

    return ci_list, di_list

def generate_all_a(q, a_list):
    a_tol = []
    for i in range(q+1):
        if i < len(a_list):
            a_tol.append(a_list[i])
        else:
            idx_sym = q - i
            a_tol.append(a_list[idx_sym])

    return a_tol

def generate_all_b(q, b_list):
    b_tol = []
    
    for i in range(q):
        if i < len(b_list):
            b_tol.append(b_list[i])
        else:
            idx_sym = q - i - 1
            b_tol.append(b_list[idx_sym])

    return b_tol

def verlet_leapforg_order2():
    return [1/2], [1]

def omelyan_order2():
    q = 2
    
    a1 = 0.1931833275037836
    a2 = 1 - 2*a1

    b1 = 1/2

    a_tol = generate_all_a(q, [a1, a2])
    b_tol = generate_all_b(q, [b1])
    
    return a_tol, b_tol
    # return [a1,a2], [b1,b2]

def suzuki_order4():
    q = 5
    
    a1 = 0.2072453858971879
    a2 = 0.4144907717943757
    a3 = 0.5 - a1 - a2
        
    b1 = 0.4144907717943757
    b2 = 0.4144907717943757
    b3 = 1 - 2*(b1 + b2)

    a_half = [a1, a2, a3]
    b_half = [b1, b2, b3]
    
    a_tol = generate_all_a(q, a_half)
    b_tol = generate_all_b(q, b_half)

    return a_tol, b_tol

def blanes_moan_order4():
    q = 6
    
    a1 = 0.07920369643119569
    a2 = 0.353172906049774
    a3 = -0.0420650803577195
    a4 = 1 - 2*(a1 + a2 + a3)
    
    b1 = 0.209515106613362
    b2 = -0.143851773179818
    b3 = 0.5 - b1 - b2
    
    a_half = [a1, a2, a3, a4]
    b_half = [b1, b2, b3]
    
    a_tol = generate_all_a(q, a_half)
    b_tol = generate_all_b(q, b_half)
    
    return a_tol, b_tol

if __name__ == "__main__":
    
    # ci, di = calc_decomposition_coeff(*verlet_leapforg_order2())
    # ci, di = calc_decomposition_coeff(*omelyan_order2())
    # ci, di = calc_decomposition_coeff(*suzuki_order4())
    # ci, di = calc_decomposition_coeff(*blanes_moan_order4())
    
    
    # print(ci)
    # print(di)
    omelyan_order2()
    # blanes_moan_order4()