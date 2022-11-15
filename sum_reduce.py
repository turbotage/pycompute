import numpy as np
import cupy as cp

import cuda.cuda_program as cuda_cp
from cuda.cuda_program import CudaTensor, CudaFunction
from cuda.symbolic import EvalJacHes
from cuda.linalg import SumEveryNUptoM

import time
import torch

ndata = 21
nparam = 4
nconst = 1

batch_size = 1000000

Nelem = batch_size * ndata

#def from_cu_tensor_flatten(t: CudaTensor, rand=False):
#    if rand:
#        return cp.random.rand(t.shape[0],t.shape[1], dtype=t.dtype)
#    else:
#        return cp.empty(shape=(t.shape[0],t.shape[1]), dtype=t.dtype)

def from_cu_tensor(t: CudaTensor, rand=False):
    if rand:
        return cp.random.rand(*(t.shape), dtype=t.dtype)
    else:
        return cp.empty(shape=tuple(t.shape), dtype=t.dtype)

sred = CudaTensor([1, Nelem], cp.float32)
sred_t = from_cu_tensor(sred, True)

n = 2
m = 7

sr = SumEveryNUptoM(sred, n, m)

ejh_code = sr.get_kernel_code()

#print(nlsq_code)
with open("bk_sred.cu", "w") as f:
    f.write(ejh_code)

ejh_kernel = cp.RawKernel(ejh_code, sr.get_funcid())
print(ejh_kernel.attributes)

NthreadPerM = np.ceil(m / (2*n))
Nthreads = 128
blockSize = max(np.ceil(Nelem / Nthreads / NthreadPerM).astype(int), 1)

ns = [0, blockSize - 1, blockSize, Nelem - 1]


sred_t2 = sred_t.copy()
print('Before kernel')
start = time.time()
for i in range(0,1):
    #nlsq_kernel((blockSize,), (Nthreads,), (pars_t, consts_t, data_t, res_t, jac_t, hes_t, batch_size))
    ejh_kernel((blockSize,), (Nthreads,), (sred_t, Nelem))
    #nlsq_kernel((blockSize,), (Nthreads,), (pars_t, consts_t, data_t, weights_t, cp.float32(0.0), res_t, jac_t, hes_t, lhes_t, batch_size))
cp.cuda.stream.get_current_stream().synchronize()
end = time.time()
print('After kernel')
print('It took: ' + str(end - start) + ' s')

for ni in ns:
    print('Show Iter: ')
    print(sred_t2[:,ni:(ni+m)])
    print(sred_t[:,ni:(ni+m)])



