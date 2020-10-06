import numpy as np

def onehotCategorical(req, limit, store=0):
    arr = np.zeros((limit,))
    arr[req-1] = 1.
    if store == 1: # encode Store
        arr = np.delete(arr, [290, 621, 878])
    return arr
