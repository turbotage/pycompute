

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

N1 = 256
N2 = 256
N3 = 256
NX = N1*N2*N3
NF = 200000

coord = cp.empty((3,NF), dtype=cp.float32)
coord[0,:] = cp.random.uniform(-cp.pi, cp.pi, NF).astype(cp.float32)
coord[1,:] = cp.random.uniform(-cp.pi, cp.pi, NF).astype(cp.float32)
coord[2,:] = cp.random.uniform(-cp.pi, cp.pi, NF).astype(cp.float32)
coord = cp.transpose(coord, (1,0))

p = cp.empty((1,N1,N2,N3), dtype=cp.complex64)
p = (cp.random.standard_normal((1,N1,N2,N3)) + 1j * cp.random.standard_normal((1,N1,N2,N2))).astype(cp.complex64)

nufft = fulinops.NUFFT((1,N1,N2,N3), coord, oversamp=1.5, width=4)
nufft_adj = fulinops.NUFFTAdjoint((1,N1,N2,N3), coord, oversamp=1.5, width=4)

cp.cuda.get_current_stream().synchronize()


start = time.time()
p_out_1 = nufft.apply(p)
p_out_1 = nufft_adj.apply(p_out_1)
cp.cuda.get_current_stream().synchronize()
end = time.time()
duration = end - start

dur_mean = 0
for i in range(10):
	start = time.time()
	p_out_1 = nufft.apply(p)
	p_out_1 = nufft_adj.apply(p_out_1)
	cp.cuda.get_current_stream().synchronize()
	end = time.time()
	duration = end - start
	dur_mean += duration
	
dur_mean /= 10
print(dur_mean)
