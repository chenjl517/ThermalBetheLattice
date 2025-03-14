import numpy as np

def seq_kron(*args):
    res = args[0]
    for i in range(1, len(args)):
        res = np.kron(res, args[i])
    return res
