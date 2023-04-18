
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

N1 = 3
N2 = 5
N3 = 4
NX = N1*N2*N3
NF = NX

coord = np.empty((3,NF), dtype=np.float32)
l = 0
for x in range(N1):
	for y in range(N2):
		for z in range(N3):
			kx = -np.pi + x * 2 * np.pi / N1
			ky = -np.pi + y * 2 * np.pi / N2
			kz = -np.pi + z * 2 * np.pi / N3

			coord[0,l] = kx
			coord[1,l] = ky
			coord[2,l] = kz
			l += 1

#coord = np.transpose(coord, (1,0))

c = np.ones((N1,N2,N3), dtype=cp.complex64)

nuftt2_out = finufft.nufft3d2(coord[0,:], coord[1,:], coord[2,:], c)

print('real: ')
print(np.real(nuftt2_out))
print('imag: ')
print(np.imag(nuftt2_out))