import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from testrun import test_runner
test_runner()

import numpy as np
import cupy as cp

import pycompute.cuda.cuda_program as cuda_cp
from pycompute.cuda.solver import GMW81Solver

ndim = 4
nmat = round(ndim*(ndim+1)/2)
batch_size=10

gmw81solcu = GMW81Solver(ndim, cp.float32, True)

mat = 10*cp.random.rand(nmat,batch_size).astype(cp.float32)
rhs = 10*cp.random.rand(ndim, batch_size).astype(cp.float32)
for i in range(0,3):
    #mat *= (10*cp.random.rand()) * cp.random.rand(nmat,batch_size).astype(cp.float32)
    #rhs *= (10*cp.random.rand()) * cp.random.rand(ndim, batch_size).astype(cp.float32)
    mat *= mat
    rhs *= rhs

#mat *= mat
#rhs *= rhs

sol = cp.empty((ndim,batch_size)).astype(cp.float32)

k = 0
for i in range(0,ndim):
    for j in range(0,i+1):
        if i == j:
            mat[k,:] *= 100
        k += 1

mat_old = mat.copy()


gmw81solcu.run(mat, rhs, sol)

for i in range(0,batch_size):
    A = cuda_cp.compact_to_full(mat_old[:,i])
    L, D = cuda_cp.compact_to_LD(mat[:,i])
    b = rhs[:,i]

    try:
        np.linalg.cholesky(A.get())
        print('is PD')
    except:
        print('not PD')
        
    print('old A: \n', A)
    #print('new A: \n', D)
    print('cp.linalg: ', cp.linalg.solve(A,b))
    print('GMW81:     ', sol[:,i])
    print('')


