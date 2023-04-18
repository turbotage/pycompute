
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

import finufft

N1 = 8
NX = N1
NF = NX

coord = np.empty((1,NF), dtype=np.float32)
l = 0
for x in range(N1):
	kx = -np.pi + x * 2 * np.pi / N1
	coord[0,l] = kx
	l += 1

#coord = np.transpose(coord, (1,0))

c = np.ones((N1,), dtype=cp.complex64)

nuftt2_out = finufft.nufft1d2(coord[0,:], c)

print('real: ')
print(np.real(nuftt2_out))
print('imag: ')
print(np.imag(nuftt2_out))