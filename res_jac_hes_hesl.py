import numpy as np
import cupy as cp

import cuda.cuda_program as cuda_cp
from cuda.cuda_program import CudaTensor, CudaFunction
from cuda.symbolic import EvalJacHes, ResJacGradHesHesl, NLSQ_LM
from cuda.solver import GMW81Solver

import time
import torch

ndata = 21
nparam = 4
nconst = 1

batch_size = 10000

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
pars_t = from_cu_tensor(pars, rand=True)

consts = CudaTensor([nconst, Nelem], cp.float32)
consts_t = from_cu_tensor(consts, rand=True)

data = CudaTensor([1, Nelem], cp.float32)
data_t = from_cu_tensor(data, rand=True)

lam = CudaTensor([1, Nelem], cp.float32)
lam_t = from_cu_tensor(lam, ones=True)
lam_t = 5*lam_t

step = CudaTensor([4, batch_size], cp.float32)
step_t = from_cu_tensor(step)

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

rjghhl = ResJacGradHesHesl(expr, pars_str, consts_str, cp.float32)
rjghhl_code = cuda_cp.code_gen_walking(rjghhl, "")

with open("bk_res_jac_grad_hes_hesl.cu", "w") as f:
    f.write(rjghhl.build())

gmw81sol = GMW81Solver(nparam, cp.float32)
gmw81sol_code = cuda_cp.code_gen_walking(gmw81sol, "")

with open("bk_gmw81sol.cu", "w") as f:
    f.write(gmw81sol.build())


print('Before kernel')
start = time.time()

hsum = None
hlsum = None
gsum = None

for i in range(0,1):
    rjghhl.run(pars_t, consts_t, data_t, lam_t, res_t, jac_t, grad_t, hes_t, hesl_t, Nelem)
    (hsum, hlsum, gsum) = NLSQ_LM.compact_rjghhl(ndata, grad_t, hes_t, hesl_t)
    cp.cuda.stream.get_current_stream().synchronize()
    print(hlsum[:,0])
    print(hlsum.ravel()[batch_size])
    print(hlsum[:,1])
    #print(gsum[:,0])
    #print(step_t[:,0])
    print('\nbefore run:')
    gmw81sol.run(hlsum, -gsum, step_t, batch_size)
    cp.cuda.stream.get_current_stream().synchronize()
    print('\nafter run')
    print(hlsum[:,0])
    print(hlsum.ravel()[batch_size])
    print(hlsum[:,1])
    #print(gsum[:,0])
    #print(step_t[:,0])

cp.cuda.stream.get_current_stream().synchronize()
end = time.time()
print('It took: ' + str(end - start) + ' s')

ns = [0, batch_size - 1]
printing=False
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
        st = step_t[:,ni]

        print(pt)
        print(ct)
        print(dt)
        print(rt)
        print(gt)
        print(hlt)
        print(ht)
        print(st)


