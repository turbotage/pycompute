import numpy as np
import cupy as cp

import cuda.cuda_program as cuda_cp
from cuda.cuda_program import CudaTensor, CudaFunction
from cuda.symbolic import EvalJacHes

import time
import torch

ndata = 21
nparam = 4
nconst = 1

batch_size = 1000000

Nelem = batch_size * ndata

def from_cu_tensor(t: CudaTensor, rand=False):
    if rand:
        return cp.random.rand(*(t.shape), dtype=t.dtype)
    else:
        return cp.empty(shape=tuple(t.shape), dtype=t.dtype)

pars = CudaTensor([4, Nelem], cp.float32)
pars_t = from_cu_tensor(pars, True)

consts = CudaTensor([nconst, Nelem], cp.float32)
consts_t = from_cu_tensor(consts, True)

#data = CudaTensor([1, Nelem], cp.float32)
#data_t = from_cu_tensor(data, True)


eval = CudaTensor([1, Nelem], cp.float32)
eval_t = from_cu_tensor(eval)

jac = CudaTensor([nparam, Nelem], cp.float32)
jac_t = from_cu_tensor(jac)

hes = CudaTensor([round(nparam*(nparam+1)/2), Nelem], cp.float32)
hes_t = from_cu_tensor(hes)

expr = 'S0*(f*exp(-b*D_1)+(1-f)*exp(-b*D_2))'
pars_str = ['S0', 'f', 'D_1', 'D_2']
consts_str = ['b']

ejh_rjh = EvalJacHes(expr, pars_str, consts_str, pars, consts, eval, jac, hes)

ejh_code = ejh_rjh.get_kernel_code()

#print(nlsq_code)
with open("bk_eval_jac_hes.cu", "w") as f:
    f.write(ejh_code)

ejh_kernel = cp.RawKernel(ejh_code, ejh_rjh.get_funcid())
print(ejh_kernel.attributes)

Nthreads = 32
blockSize = max(np.ceil(Nelem / Nthreads).astype(int), 1)

ns = [0, blockSize - 1, blockSize, Nelem - 1]

print('Before kernel')
start = time.time()
for i in range(0,10000):
    #nlsq_kernel((blockSize,), (Nthreads,), (pars_t, consts_t, data_t, res_t, jac_t, hes_t, batch_size))
    ejh_kernel((blockSize,), (Nthreads,), (pars_t, consts_t, eval_t, jac_t, hes_t, Nelem))
    #nlsq_kernel((blockSize,), (Nthreads,), (pars_t, consts_t, data_t, weights_t, cp.float32(0.0), res_t, jac_t, hes_t, lhes_t, batch_size))
cp.cuda.stream.get_current_stream().synchronize()
end = time.time()
print('After kernel')
print('It took: ' + str(end - start) + ' s')

#for ni in ns:
#    print('Show Iter: ')
#    pt = pars_t[:,ni]
#    ct = consts_t[:,ni]
#    #dt = data_t[:,ni]
#    rt = eval_t[:,ni]
#    jt = jac_t[:,ni]
#    ht = hes_t[:,ni]
#
#    print(pt)
#    print(ct)
#    #print(dt)
#    print(rt)
#    print(jt)
#    print(ht)
#
#    print(pt[0]*(pt[1]*np.exp(-ct[0]*pt[2])+(1-pt[1])*np.exp(-ct[0]*pt[3])))


