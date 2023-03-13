

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


N1 = 140
N2 = 140
N3 = 140
NX = N1*N2*N3
NF = 80000



coord = cp.empty((3,NF), dtype=cp.float32)
coord[0,:] = cp.random.uniform(-cp.pi, cp.pi, NF).astype(cp.float32)
coord[1,:] = cp.random.uniform(-cp.pi, cp.pi, NF).astype(cp.float32)
coord[2,:] = cp.random.uniform(-cp.pi, cp.pi, NF).astype(cp.float32)
coord = cp.transpose(coord, (1,0))
print(coord.shape)


p = cp.empty((1,N1,N2,N3), dtype=cp.complex64)
p = (cp.random.standard_normal((1,N1,N2,N3)) + 1j * cp.random.standard_normal((1,N1,N2,N2))).astype(cp.complex64)
print(p.shape)

P = linop.Multiply((1,N1,N2,N3), p)
P.apply(p)






#p_out_1 = fourier.nufft(p,coord,2.0,4)
#p_out_1 = fourier.nufft_adjoint(p_out_1, coord, (1,256,256,256), 2.0, 4)

nufft = fulinops.NUFFT((1,N1,N2,N3), coord, oversamp=2.0, width=4)
nufft_adj = fulinops.NUFFTAdjoint((1,N1,N2,N3), coord, oversamp=2.0, width=4)

cp.cuda.get_current_stream().synchronize()

start = time.time()
p_out_1 = nufft.apply(p)
p_out_1 = nufft_adj.apply(p_out_1)
cp.cuda.get_current_stream().synchronize()
end = time.time()
print(end - start)

nufft2 = fulinops.NormalNUFFT((1,N1,N2,N3), coord, 2.0, 4)

cp.cuda.get_current_stream().synchronize()

start = time.time()
p_out_2 = nufft2.apply(p)
cp.cuda.get_current_stream().synchronize()
end = time.time()
print(end - start)


pdiff = cp.linalg.norm(p_out_1 - p_out_2) / cp.linalg.norm(p_out_1)
print(pdiff)

