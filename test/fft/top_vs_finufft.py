
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

N1 = 5
N2 = 3
N3 = 7
NX = N1*N2*N3
NF = NX

coord = np.empty((3,NF), dtype=np.float32)
l = 0
for x in range(N1):
	for y in range(N2):
		for z in range(N3):
			kx = -np.pi + x * 2.0 * np.pi / N1
			ky = -np.pi + y * 2.0 * np.pi / N2
			kz = -np.pi + z * 2.0 * np.pi / N3

			coord[0,l] = kx
			coord[1,l] = ky
			coord[2,l] = kz

			l += 1

#coord[0,:] = -np.pi + 2 * np.pi * np.random.rand(NF).astype(np.float32)
#coord[1,:] = -np.pi + 2 * np.pi * np.random.rand(NF).astype(np.float32)
#coord[2,:] = -np.pi + 2 * np.pi * np.random.rand(NF).astype(np.float32)

coord = np.transpose(coord, (1,0))

#c = np.ones((1,N1,N2,N3), dtype=cp.complex64)

sp_coords = cp.array(coord)
sp_coords[:,0] *= (N1 / (2*cp.pi))
sp_coords[:,1] *= (N2 / (2*cp.pi))
sp_coords[:,2] *= (N3 / (2*cp.pi))
#sp_temp = sp_coords[:,0]
#sp_coords[:,0] = sp_coords[:,2]
#sp_coords[:,2] = sp_temp

psf1 = fourier.toeplitz_psf(sp_coords, (1,N1,N2,N3), oversamp=2.0, width=4).get()
#psf1 = np.fft.ifftn(psf1)
psf1_f = psf1.flatten()

unity_vector = np.zeros((2*N1,2*N2,2*N3), dtype=cp.complex64)
#unity_vector[0,0,0] = 1
unity_vector[N1,N2,N3] = 1.0
#unity_vector = np.fft.ifftshift(unity_vector)
psf2 = finufft.nufft3d2(coord[:,0], coord[:,1], coord[:,2], unity_vector)
psf2 = finufft.nufft3d1(coord[:,0], coord[:,1], coord[:,2], psf2, (2*N1,2*N2,2*N3))
#psf2 = np.fft.ifftshift(psf2)
psf2 = np.fft.ifftshift(psf2)
psf2 = np.fft.fftn(psf2)
#psf2 = np.fft.ifftshift(psf2)

psf2_f = psf2.flatten()

# plt.figure()
# plt.plot(np.abs(psf1_f), 'r-')
# plt.plot(np.abs(psf2_f), 'g-')
# plt.legend(['nufft1', 'nufft2'])
# plt.show()

# plt.figure()
# plt.plot(np.real(psf1_f), 'r-')
# plt.plot(np.real(psf2_f), 'g-')
# plt.legend(['nufft1', 'nufft2'])
# plt.show()

# plt.figure()
# plt.plot(np.imag(psf1_f), 'r-')
# plt.plot(np.imag(psf2_f), 'g-')
# plt.legend(['nufft1', 'nufft2'])
# plt.show()

#p = (np.random.rand(1,N1,N2,N3).astype(np.float32) + 1j*np.random.rand(1,N1,N2,N3).astype(np.float32)).astype(np.complex64)
p = np.ones((1,N1,N2,N3), dtype=np.complex64)

nufft_out1 = p.squeeze(0)
nufft_out1 = finufft.nufft3d2(coord[:,0], coord[:,1], coord[:,2], nufft_out1)
nufft_out1 = finufft.nufft3d1(coord[:,0], coord[:,1], coord[:,2], nufft_out1, (N1,N2,N3))
#nufft_out1 = np.fft.fftshift(nufft_out1)
nufft_out1 = nufft_out1.flatten()

diag = psf2
#diag = np.fft.ifftshift(diag)

nufft_out2 = p
nufft_out2 = np.fft.fftn(nufft_out2, s=(2*N1,2*N2,2*N3), axes=(1,2,3))
nufft_out2 = diag * nufft_out2
nufft_out2 = np.fft.ifftn(nufft_out2, s=(2*N1,2*N2,2*N3), axes=(1,2,3))
#nufft_out2 = np.fft.ifftn(nufft_out2, s=(N1,N2,N3), axes=(1,2,3))
#nufft_out2 = nufft_out2[:,(N1 // 2):((N1 // 2) + N1),(N2 // 2):((N2 // 2) + N2),(N3 // 2):((N3 // 2) + N3)]
nufft_out2 = nufft_out2[:,:N1,:N2,:N3]

#nufft_out2 = np.fft.ifftshift(nufft_out2)
nufft_out2 = nufft_out2.flatten()

#nufft_out1 = nufft_out1[:100]
#nufft_out2 = nufft_out2[:100]

plt.figure()
plt.plot(np.abs(nufft_out1), 'r-')
plt.plot(np.abs(nufft_out2), 'g-')
plt.legend(['nufft1', 'nufft2'])
plt.show()

plt.figure()
plt.plot(np.real(nufft_out1), 'r-')
plt.plot(np.real(nufft_out2), 'g-')
plt.legend(['nufft1', 'nufft2'])
plt.show()

plt.figure()
plt.plot(np.imag(nufft_out1), 'r-')
plt.plot(np.imag(nufft_out2), 'g-')
plt.legend(['nufft1', 'nufft2'])
plt.show()


print(np.real(nufft_out1).flatten())
print(np.imag(nufft_out1).flatten())
print(np.abs(nufft_out1).flatten())