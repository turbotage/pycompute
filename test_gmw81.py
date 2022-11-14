import numpy as np
import cupy as cp

import cuda.cuda_program as cuda_cp
from cuda.cuda_program import CudaTensor, CudaFunction
from cuda.solver import GMW81Solver

import time

import torch

batch_size = 1000000
#cpu_mat = np.random.rand(batch_size, 4, 4, dtype=np.float32)
cpu_mat = torch.rand((batch_size, 4, 4), dtype=torch.float32)
cpu_mat = torch.bmm(cpu_mat.transpose(1,2), cpu_mat)
cpu_mat = cpu_mat.numpy()

cpu_rhs = np.random.rand(batch_size, 4, 1).astype(np.float32)
cpu_sol = np.zeros((batch_size, 4, 1), dtype=np.float32)

gpu_mat = cp.asarray(cpu_mat.copy())
gpu_rhs = cp.asarray(cpu_rhs.copy())
gpu_sol = cp.asarray(cpu_sol.copy())

mat = CudaTensor([batch_size, 4, 4], cp.float32)
rhs = CudaTensor([batch_size, 4, 1], cp.float32)
sol = CudaTensor([batch_size, 4, 1], cp.float32)

gmw81solver = GMW81Solver(mat, rhs, sol)

gmw81_code = cuda_cp.code_gen_walking(gmw81solver, "")

gmw81_code += gmw81solver.get_batched_kernel()


print(gmw81_code)
with open("bk_gmw81.cu", "w") as f:
    f.write(gmw81_code)

gmw81_module = cp.RawModule(code=gmw81_code)
gmw81_kernal = gmw81_module.get_function('bk_' + gmw81solver.get_funcid())


print(gmw81_kernal.attributes)

Nthreads = 256
blockSize = max(np.ceil(batch_size / Nthreads).astype(int), 1)

print('blockSize: ' + str(blockSize))


ns = [0, blockSize - 1, blockSize, batch_size - 1]

for ni in ns:
    temp_sol = cp.linalg.solve(gpu_mat[ni,:], gpu_rhs[ni,:])
    print(temp_sol)

print('Before kernel')

start = time.time()
gmw81_kernal((blockSize,), (Nthreads,), (gpu_mat, gpu_rhs, gpu_sol, batch_size))
cp.cuda.stream.get_current_stream().synchronize()
end = time.time()

print('After kernel')
print('It took: ' + str(end - start) + ' s')

for ni in ns:
    print(gpu_sol[ni,:])
