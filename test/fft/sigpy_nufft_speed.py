
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

import matplotlib.pyplot as plt

NX = 256
NF = 200000

widthw = 8

input = (cp.random.rand(1,NX,NX,NX) + 1j*cp.random.rand(1,NX,NX,NX)).astype(cp.complex64)
coord = -0.5*NX + cp.random.rand(3,NF).astype(cp.float32) * NX
coord = cp.transpose(coord, (1,0))

output1 = fourier.nufft(input, coord, oversamp=1.5, width=widthw)
output2 = fourier.nufft_adjoint(output1, coord, (1,NX,NX,NX), oversamp=1.5, width=widthw)

cp.cuda.get_current_stream().synchronize()

start = time.time()

for i in range(40):
	output1 = fourier.nufft(input, coord, oversamp=1.5, width=widthw, center=False)
	output2 = fourier.nufft_adjoint(output1, coord, (1,NX,NX,NX),oversamp=1.5, width=widthw, center=False)	

cp.cuda.get_current_stream().synchronize()

end = time.time()

print(end - start)