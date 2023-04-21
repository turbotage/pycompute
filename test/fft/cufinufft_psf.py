
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
N2 = 2
N3 = 2
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

coord[0,:] = -np.pi + 2 * np.pi * np.random.rand(NF).astype(np.float32) / N1
coord[1,:] = -np.pi + 2 * np.pi * np.random.rand(NF).astype(np.float32) / N2
coord[2,:] = -np.pi + 2 * np.pi * np.random.rand(NF).astype(np.float32) / N3

coord = np.transpose(coord, (1,0))

#c = np.ones((1,N1,N2,N3), dtype=cp.complex64)

sp_coords = cp.array(coord)
sp_coords[:,0] *= (N1 / (2*cp.pi))
sp_coords[:,1] *= (N2 / (2*cp.pi))
sp_coords[:,2] *= (N3 / (2*cp.pi))
#sp_temp = sp_coords[:,0]
#sp_coords[:,0] = sp_coords[:,2]
#sp_coords[:,2] = sp_temp

psf1 = fourier.toeplitz_psf(sp_coords, (1,N1,N2,N3), oversamp=2.0, width=8).get()
#psf1 = np.fft.ifftn(psf1)

unity_vector = np.zeros((2*N1,2*N2,2*N3), dtype=cp.complex64)
unity_vector[0,0,0] = 1

print(unity_vector)

unity_vector = np.fft.ifftshift(unity_vector)
nuftt2_out = finufft.nufft3d2(coord[:,0], coord[:,1], coord[:,2], unity_vector) / np.sqrt(NX)
nufft1_out = finufft.nufft3d1(coord[:,0], coord[:,1], coord[:,2], nuftt2_out, (2*N1,2*N2,2*N3)) / np.sqrt(NX)
psf2 = np.fft.fftshift(nufft1_out)
#psf2 = nufft1_out
psf2 = np.fft.fftn(psf2)
psf2 = np.fft.fftshift(psf2)

psf1 = psf1.flatten()
psf2 = psf2.flatten()


plt.figure()
plt.plot(np.abs(psf1), 'r-')
plt.plot(np.abs(psf2), 'g-')
plt.show()

plt.figure()
plt.plot(np.real(psf1), 'r-')
plt.plot(np.real(psf2), 'g-')
plt.show()

plt.figure()
plt.plot(np.imag(psf1), 'r-')
plt.plot(np.imag(psf2), 'g-')
plt.show()



