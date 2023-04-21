
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

N1 = 64
NX = N1
NF = NX

coord = np.empty((3,NF), dtype=np.float32)
l = 0
for x in range(N1):
	kx = -np.pi + x * 2 * np.pi / N1
	coord[0,l] = kx
	l += 1
coord[0,:] = np.random.uniform(-np.pi, np.pi, NF).astype(np.float32)

coord = np.transpose(coord, (1,0))

nufft_matrix = np.empty((NF,N1), dtype=np.complex64)
for i in range(0,NF):
	for k in range(0,N1):
		nufft_matrix[i,k] = np.exp(-1j*coord[i,0]*k).squeeze()

ATA = np.conjugate(np.transpose(nufft_matrix, (1,0))) @ nufft_matrix
psf1 = np.empty((2*NX,), dtype=np.complex64)
psf1[0:N1] = ATA[:,0]
psf1[(N1+1):] = np.flip(ATA[0,1:])
psf1[N1] = psf1[0]
psf1 = np.fft.fft(psf1)

unity_vector = np.zeros((2*N1,), dtype=cp.complex64)
unity_vector[0] = 1
#unity_vector = np.fft.ifftshift(unity_vector)
psf2 = finufft.nufft1d2(coord[:,0], unity_vector)
psf2 = finufft.nufft1d1(coord[:,0], psf2, (2*N1,))
#psf2 = np.fft.fftshift(psf2)
#psf2 = np.fft.fftn(psf2)
#psf2 = np.fft.fftshift(psf2)

psf1_f = psf1.flatten()
psf2_f = psf2.flatten()

plt.figure()
plt.plot(np.abs(psf1_f), 'r-')
plt.plot(np.abs(psf2_f), 'g-')
plt.show()

plt.figure()
plt.plot(np.real(psf1_f), 'r-')
plt.plot(np.real(psf2_f), 'g-')
plt.show()

plt.figure()
plt.plot(np.imag(psf1_f), 'r-')
plt.plot(np.imag(psf2_f), 'g-')
plt.show()

p = (np.random.rand(N1).astype(np.float32) + 1j*np.random.rand(N1).astype(np.float32)).astype(np.complex64)

nufft_out1 = p
nufft_out1 = finufft.nufft1d2(coord[:,0], nufft_out1)
nufft_out1 = finufft.nufft1d1(coord[:,0], nufft_out1, (N1,))
#nufft_out1 = np.fft.fftshift(nufft_out1)
nufft_out1 = nufft_out1.flatten()

nufft_out2 = p
nufft_out2 = np.fft.fft(nufft_out2, 2*N1)
nufft_out2 = psf1 * nufft_out2
#nufft_out2 = cp.fft.ifftshift(cp.array(psf2)) * nufft_out2
#nufft_out2 = cp.fft.fftshift(cp.array(psf2)) * nufft_out2
nufft_out2 = np.fft.ifft(nufft_out2, N1)
#nufft_out2 = fourier.fft(cp.array(p), (1,2*N1,2*N2,2*N3),(1,2,3))
#nufft_out2 = cp.array(psf1) * nufft_out2
#nufft_out2 = fourier.ifft(nufft_out2, (1,N1,N2,N3),(1,2,3))
#nufft_out2 = nufft_out2.get().flatten()

nufft_out1 = nufft_out1[:100]
nufft_out2 = nufft_out2[:100]

plt.figure()
plt.plot(np.abs(nufft_out1), 'r-')
plt.plot(np.abs(nufft_out2), 'g-')
#plt.plot(np.abs(nufft_out3), 'b-')

plt.legend(['nufft1', 'nufft2'])
plt.show()

plt.figure()
plt.plot(np.real(nufft_out1), 'r-')
plt.plot(np.real(nufft_out2), 'g-')
#plt.plot(np.real(nufft_out3), 'b-')

plt.legend(['nufft1', 'nufft2'])
plt.show()

plt.figure()
plt.plot(np.imag(nufft_out1), 'r-')
plt.plot(np.imag(nufft_out2), 'g-')
#plt.plot(np.imag(nufft_out3), 'b-')

plt.legend(['nufft1', 'nufft2'])
plt.show()



