
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

N1 = 2
NX = N1*N1*N1
NF = N1*N1*N1

coord = np.empty((3,NF), dtype=np.float32)
l = 0
for x in range(N1):
	for y in range(N1):
		for z in range(N1):
			kx = -np.pi + x * 2 * np.pi / N1
			ky = -np.pi + y * 2 * np.pi / N1
			kz = -np.pi + z * 2 * np.pi / N1

			coord[0,l] = kx
			coord[1,l] = ky
			coord[2,l] = kz

			l += 1

coord = np.random.rand(3,NF).astype(np.float32)
coord = np.transpose(coord, (1,0))

c = np.ones((1,N1,N1,N1), dtype=cp.complex64)

sp_coords = cp.array(coord) * (N1 / (2*cp.pi))
#sp_temp = sp_coords[:,0]
#sp_coords[:,0] = sp_coords[:,2]
#sp_coords[:,2] = sp_temp


f1 = np.sqrt(NX) * fourier.nufft(cp.array(c), sp_coords, oversamp=2.0, width=4, center=True).get().squeeze(0)
f2 = finufft.nufft3d2(coord[:,0], coord[:,1], coord[:,2], c.squeeze(0))
f3 = np.fft.fftn(c).flatten()

#print(np.abs(f1))
#print(np.abs(f2))
#print(np.abs(f3))

plt.figure()
plt.plot(np.abs(f1), 'r-')
plt.plot(np.abs(f2), 'g-')
#plt.plot(np.abs(f3), 'b-')
plt.show()

plt.figure()
plt.plot(np.real(f1), 'r-')
plt.plot(np.real(f2), 'g-')
#plt.plot(np.real(f3), 'b-')
plt.show()

plt.figure()
plt.plot(np.imag(f1), 'r-')
plt.plot(np.imag(f2), 'g-')
#plt.plot(np.imag(f3), 'b-')
plt.show()
