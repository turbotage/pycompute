

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

NX = 6
NF = 6

coord = cp.empty((1,NF), dtype=cp.float32)
#coord[0,:] = cp.random.uniform(-cp.pi, cp.pi, NF).astype(cp.float32)
for i in range(NF):
	coord[0,i] = cp.pi*i/NF
print(coord)

print('coord')

#coord[1,:] = cp.random.uniform(-cp.pi, cp.pi, NF).astype(cp.float32)
#coord[2,:] = cp.random.uniform(-cp.pi, cp.pi, NF).astype(cp.float32)
coord = cp.transpose(coord, (1,0))

nufft_matrix = cp.empty((NF,NX), dtype=cp.complex64)

for i in range(0,NF):
	for j in range(0,NX):
		nufft_matrix[i,j] = cp.exp(-1j*coord[i]*j).squeeze()

#print(nufft_matrix)

#p = (cp.random.uniform(0,1, size=(NX,)) + 1j*cp.random.uniform(0,1, size=(NX,))).astype(cp.complex64)
#p = cp.empty((NX,), dtype=cp.complex64)
p = cp.arange(NX).astype(cp.complex64)
p = 0.5*cp.ones((NX,)) + 0.4*cp.sin(p * (cp.pi*2/NF)) + 0.3*cp.sin(p * (cp.pi*4/NF))
#print(p.get())

#p[0] = 0.5 + 1j
#p[1] = 0 #0.7 - 0.7j
#p[2] = 0 #0.3 + 0.3j
#p[3] = 0 #0.2 - 0.1j

pexpand = cp.zeros((2*NX,), dtype=cp.complex64)
pexpand[0:NX] = p

pnufft = p
#pnufft = cp.fft.ifftshift(pnufft)
pnufft = cp.sqrt(NX) * fourier.nufft(pnufft, coord, oversamp=4.0, width=8, center=True)

print(cp.real(pnufft).get())

pmynufft = p
#pmynufft = cp.fft.fftshift(pmynufft)
#pmynufft = cp.fft.ifftshift(pmynufft)
pmynufft = nufft_matrix @ pmynufft
#pmynufft = cp.fft.ifftshift(pmynufft)
#pmynufft = cp.fft.fftshift(pmynufft)
print(cp.real(pmynufft).get())

pout = cp.transpose(nufft_matrix, (1,0)).conj() @ nufft_matrix @ p

#print(pout.get())

psf = fourier.toeplitz_psf(coord, shape=(NX,), oversamp=4.0, width=2)

#pexpout = cp.fft.ifftshift(pexpand)
pexpout = cp.fft.fft(pexpand)
#pexpout = cp.fft.fftshift(pexpand)
pexpout = cp.fft.ifftshift(psf) * pexpout
#pexpout = cp.fft.ifftshift(pexpand)
pexpout = cp.fft.ifft(pexpand)
#pexpout = cp.fft.fftshift(pexpand)

pexpout = pexpout[0:NX]
#print(pexpout.get())



