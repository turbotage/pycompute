import numpy as np
import cupy as cp

import cuda.cuda_program as cuda_cp
from cuda.cuda_program import CudaTensor, CudaFunction
from cuda.linalg import GMW81Solver

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

blockSize = np.ceil(batch_size / gmw81_kernal.attributes['max_threads_per_block'])

blockSize = max(np.ceil(np.sqrt(blockSize)).astype(int), 1)

print(gpu_mat[0,:])
print(gpu_rhs[0,:])
print(gpu_sol[0,:])

temp_sol = cp.linalg.solve(gpu_mat[0,:], gpu_rhs[0,:])
print(temp_sol)

gmw81_kernal((blockSize,), (blockSize,), (gpu_mat, gpu_rhs, gpu_sol, batch_size))

print(gpu_mat[0,:])
print(gpu_rhs[0,:])
print(gpu_sol[0,:])
