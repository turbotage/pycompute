import numpy as np
import cupy as cp

import cuda.cuda_program as cuda_cp
from cuda.cuda_program import CudaTensor, CudaFunction
from cuda.symbolic import EvalJacHes, ResJacGradHesHesl, NLSQ_LM

import time
import torch

ndata = 21
nparam = 4
nconst = 1

batch_size = 1000000

Nelem = batch_size * ndata

def from_cu_tensor(t: CudaTensor, rand=False, zeros=False, ones=False):
    if rand:
        return cp.random.rand(*(t.shape), dtype=t.dtype)
    if zeros:
        return cp.zeros(shape=tuple(t.shape), dtype=t.dtype)
    if ones:
        return cp.ones(shape=tuple(t.shape), dtype=t.dtype)

    return cp.empty(shape=tuple(t.shape), dtype=t.dtype)

pars = CudaTensor([4, Nelem], cp.float32)
pars_t = from_cu_tensor(pars, True)

consts = CudaTensor([nconst, Nelem], cp.float32)
consts_t = from_cu_tensor(consts, True)

data = CudaTensor([1, Nelem], cp.float32)
data_t = from_cu_tensor(data, True)

lam = CudaTensor([1, Nelem], cp.float32)
lam_t = from_cu_tensor(lam, ones=True)


res = CudaTensor([1, Nelem], cp.float32)
res_t = from_cu_tensor(res)

jac = CudaTensor([nparam, Nelem], cp.float32)
jac_t = from_cu_tensor(jac)

grad = CudaTensor([nparam, Nelem], cp.float32)
grad_t = from_cu_tensor(grad)

hes = CudaTensor([round(nparam*(nparam+1)/2), Nelem], cp.float32)
hes_t = from_cu_tensor(hes)

hesl = CudaTensor([round(nparam*(nparam+1)/2), Nelem], cp.float32)
hesl_t = from_cu_tensor(hesl)

expr = 'S0*(f*exp(-b*D_1)+(1-f)*exp(-b*D_2))'
pars_str = ['S0', 'f', 'D_1', 'D_2']
consts_str = ['b']

ejh_rjh = ResJacGradHesHesl(expr, pars_str, consts_str, cp.float32)

ejh_code = cuda_cp.code_gen_walking(ejh_rjh, "")

with open("bk_res_jac_grad_hes_hesl.cu", "w") as f:
    f.write(ejh_code)

ns = [0, Nelem - 1]

print('Before kernel')
start = time.time()

hsum = None
hlsum = None
gsum = None

for i in range(0,10):
    ejh_rjh.run(pars_t, consts_t, data_t, lam_t, res_t, jac_t, grad_t, hes_t, hesl_t, int(Nelem))
    #(hsum, hlsum, gsum) = NLSQ_LM.compact_rjghhl(ndata, grad_t, hes_t, hesl_t)

cp.cuda.stream.get_current_stream().synchronize()
end = time.time()
print('After kernel')
print('It took: ' + str(end - start) + ' s')

printing = False
if printing:
    for ni in ns:
        print('Show Iter: ')
        pt = pars_t[:,ni]
        ct = consts_t[:,ni]
        dt = data_t[:,ni]
        rt = res_t[:,ni]
        gt = gsum[:,ni]
        hlt = hlsum[:,ni]
        ht = hsum[:,ni]

        print(pt)
        print(ct)
        print(dt)
        print(rt)
        print(gt)
        print(hlt)
        print(ht)


