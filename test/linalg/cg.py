
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from testrun import test_runner
test_runner()

import numpy as np
import cupy as cp

import pycompute.cuda.sigpy.fourier_linops as fulinops
import pycompute.cuda.sigpy.fourier as fourier
import pycompute.cuda.sigpy.linop as linop
import pycompute.cuda.sigpy.solvers as solvers

from pycompute.cuda.sigpy.util2 import CupyLinopWrapper

import cupyx.scipy.sparse.linalg as sparse_linalg

N1 = 256
N2 = 256
N3 = 160
NX = N1*N2*N3
NF = 80000

coord = cp.empty((3,NF), dtype=cp.float32)
coord[0,:] = cp.random.uniform(-cp.pi, cp.pi, NF).astype(cp.float32)
coord[1,:] = cp.random.uniform(-cp.pi, cp.pi, NF).astype(cp.float32)
coord[2,:] = cp.random.uniform(-cp.pi, cp.pi, NF).astype(cp.float32)
coord = cp.transpose(coord, (1,0))

mps = cp.random.rand(1,N1,N2,N3).astype(cp.float32)

S = linop.Multiply(mps.shape, mps)

NTN = fulinops.NormalNUFFT((1,N1,N2,N3), coord, 2.0, 4)
lamda = 0.01

LHL =  S.H * NTN * S + lamda * linop.Identity(mps.shape)
CP_LHL = CupyLinopWrapper(LHL, dtype=cp.complex64)

x = (cp.random.rand(1,N1,N2,N3) + 1j * cp.random.rand(1,N1,N2,N3)).astype(cp.complex64)

b = LHL * x

CG = solvers.ConjugateGradient(LHL, b, cp.ones((1,N1,N2,N3), dtype=cp.complex64), tol=1e-6, max_iter=100)

cp.cuda.get_current_stream().synchronize()

start = time.time()

cg_not_done = True
while cg_not_done:
	#print('Update')
	CG.update()
	cg_not_done = not CG.done()
	if CG.not_positive_definite:
		print('Not positive definite')

cp.cuda.get_current_stream().synchronize()
end = time.time()
print(end - start)

relerr = cp.linalg.norm(CG.x - x) / cp.linalg.norm(x)
print(relerr.get())


cp.cuda.get_current_stream().synchronize()
start = time.time()

tupx = sparse_linalg.cg(CP_LHL, b.flatten())

end = time.time()
print(end - start)

relerr = cp.linalg.norm(tupx[0].reshape((1,N1,N2,N3)) - x) / cp.linalg.norm(x)
print(float(relerr.get()))



