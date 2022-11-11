import numpy as np
import cupy as cp

import cuda.cuda_program as cuda_cp
from cuda.cuda_program import CudaTensor, CudaFunction
from cuda.nlsq_sym import NLSQResJacHesLHes

import time
import torch

ndata = 21
nparam = 4
nconst = 1

batch_size = 1000000
#cpu_mat = np.random.rand(batch_size, 4, 4, dtype=np.float32)
#cpu_mat = torch.rand((batch_size, 4, 4), dtype=torch.float32)
#cpu_mat = torch.bmm(cpu_mat.transpose(1,2), cpu_mat)
#cpu_mat = cpu_mat.numpy()

#cpu_rhs = np.random.rand(batch_size, 4, 1).astype(np.float32)
#cpu_sol = np.zeros((batch_size, 4, 1), dtype=np.float32)

#gpu_mat = cp.asarray(cpu_mat.copy())
#gpu_rhs = cp.asarray(cpu_rhs.copy())
#gpu_sol = cp.asarray(cpu_sol.copy())


def from_cu_tensor(t: CudaTensor, rand=False):
    if rand:
        return cp.random.rand(*(t.shape), dtype=t.dtype)
    else:
        return cp.empty(shape=tuple(t.shape), dtype=t.dtype)

pars = CudaTensor([batch_size, nparam, 1], cp.float32)
pars_t = from_cu_tensor(pars, True)

consts = CudaTensor([batch_size, ndata, nconst], cp.float32)
consts_t = from_cu_tensor(consts, True)

data = CudaTensor([batch_size, ndata, 1], cp.float32)
data_t = from_cu_tensor(data, True)

res = CudaTensor([batch_size, ndata, 1], cp.float32)
res_t = from_cu_tensor(res)

jac = CudaTensor([batch_size, ndata, nparam], cp.float32)
jac_t = from_cu_tensor(jac)

hes = CudaTensor([batch_size, nparam, nparam], cp.float32)
hes_t = from_cu_tensor(hes)

lhes = CudaTensor([batch_size, nparam, nparam], cp.float32)
lhes_t = from_cu_tensor(lhes)

expr = 'S0*(f*exp(-b*D_1)+(1-f)*exp(-b*D_2))'
pars_str = ['S0', 'f', 'D_1', 'D_2']
consts_str = ['b']

nlsq_rjh = NLSQResJacHesLHes(expr, pars_str, consts_str, pars, consts, data, res, jac, hes, lhes)

nlsq_code = cuda_cp.code_gen_walking(nlsq_rjh, "")

nlsq_code += nlsq_rjh.get_batched_kernel()

#print(nlsq_code)
with open("bk_res_jac_hes.cu", "w") as f:
    f.write(nlsq_code)

nlsq_module = cp.RawModule(code=nlsq_code)
nlsq_kernel = nlsq_module.get_function('bk_' + nlsq_rjh.get_funcid())
#print(nlsq_kernel.attributes)

Nthreads = 32
blockSize = max(np.ceil(batch_size / Nthreads).astype(int), 1)

ns = [0, blockSize - 1, blockSize, batch_size - 1]

print('Before kernel')
start = time.time()
nlsq_kernel((blockSize,), (Nthreads,), (pars_t, consts_t, data_t, cp.float32(0.0), res_t, jac_t, hes_t, lhes_t, batch_size))
cp.cuda.stream.get_current_stream().synchronize()
end = time.time()
print('After kernel')
print('It took: ' + str(end - start) + ' s')

#for ni in ns:
#    print(res_t[ni,:])
#    print(jac_t[ni,:])
#    print(hes_t[ni,:])
#    print(lhes_t[ni,:])


