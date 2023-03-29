

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

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pycompute.plot import plot_utility as pu

NX = 12
NF = 12

coord = cp.empty((1,NF), dtype=cp.float32)
#coord[0,:] = cp.random.uniform(-cp.pi, cp.pi, NF).astype(cp.float32)
for i in range(NF):
	coord[0,i] = i - NX // 2 #2*cp.pi*i/NF
print(coord)

print('coord')

coord = cp.transpose(coord, (1,0))

#plt.plot(coord.get())

nufft_matrix = cp.empty((NF,NX), dtype=cp.complex64)

for i in range(0,NF):
	for j in range(0,NX):
		nufft_matrix[i,j] = cp.exp(-2j*cp.pi*coord[i]*j/NX).squeeze()

#print('nufft_matrix', nufft_matrix)

#p = (cp.random.uniform(0,1, size=(NX,)) + 1j*cp.random.uniform(0,1, size=(NX,))).astype(cp.complex64)
#p = cp.empty((NX,), dtype=cp.complex64)
p = cp.arange(NX).astype(cp.complex64)
p = cp.ones((NX,)) + 0.4*cp.exp(p * (cp.pi*2/NF)) + 0.3*cp.exp(p * (cp.pi*4/NF))
print('p: ', p.get())
#plt.plot(p.get())

#p[0] = 0.5 + 1j
#p[1] = 0 #0.7 - 0.7j
#p[2] = 0 #0.3 + 0.3j
#p[3] = 0 #0.2 - 0.1j

pexpand = cp.zeros((2*NX,), dtype=cp.complex64)
pexpand[0:NX] = p

pnufft = p
#pnufft = cp.fft.ifftshift(pnufft)
pnufft = cp.sqrt(NX) * fourier.nufft(pnufft, coord, oversamp=4.0, width=8, center=True)

#plt.plot(cp.real(pnufft).get())
plt.plot(cp.imag(pnufft).get())
#plt.plot(cp.abs(pnufft).get())

print('interp_nufft: ', cp.real(pnufft).get())

shift_vector = cp.ones((NX,), dtype=cp.complex64)
shift_vector *= cp.exp(-1j*cp.pi/(NX)) ** cp.arange(NX) # cp.exp(-1j*cp.pi / (NX - 1)) * cp.arange(NX)
#print(shift_vector)

pmynufft = p
#pmynufft = p * shift_vector
#pmynufft = cp.fft.fftshift(pmynufft)
pmynufft = cp.fft.ifftshift(pmynufft)
pmynufft = nufft_matrix @ pmynufft
#pmynufft = cp.fft.ifftshift(pmynufft)
#pmynufft = cp.fft.fftshift(pmynufft)
#pmynufft = p * shift_vector

ratio = pmynufft / pnufft
print(ratio.get())

#plt.plot(cp.real(pmynufft).get())
plt.plot(cp.imag(pmynufft).get())
#plt.plot(cp.abs(pmynufft).get())

print('matrix_nufft: ', cp.real(pmynufft).get())


plt.show()









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



