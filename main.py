import numpy as np
import cupy as cp

import cuda.cuda_program as cuda_cp
from cuda.cuda_program import CudaTensor, CudaFunction
from cuda.linalg import GMW81Solver

mat = CudaTensor([100, 4, 4], cp.float32)
rhs = CudaTensor([100, 4, 1], cp.float32)
sol = CudaTensor([100, 4, 1], cp.float32)

gmw81solver = GMW81Solver(mat, rhs, sol)

code = cuda_cp.code_gen_walking(gmw81solver, "")

print(code)

