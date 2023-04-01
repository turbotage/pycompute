

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

NX = 32
NF = 32

coord = cp.empty((1,NF), dtype=cp.float32)
#coord[0,:] = cp.random.uniform(-cp.pi, cp.pi, NF).astype(cp.float32)
for i in range(NF):
	coord[0,i] = i - NX // 2 #2*cp.pi*i/NF
#print(coord)

#print('coord')

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
##print('p: ', p.get())
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

#print('interp_nufft: ', cp.real(pnufft).get())

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

#ratio = pmynufft / pnufft
#print(ratio.get())

# plt.figure()
# plt.plot(cp.real(pmynufft).get())
# plt.plot(cp.real(pnufft).get())
# plt.legend(['my_nufft', 'nufft'])
# plt.show()

# plt.figure()
# plt.plot(cp.imag(pmynufft).get())
# plt.plot(cp.imag(pnufft).get())
# plt.legend(['my_nufft', 'nufft'])
# plt.show()

# plt.figure()
# plt.plot(cp.abs(pmynufft).get())
# plt.plot(cp.abs(pnufft).get())
# plt.legend(['my_nufft', 'nufft'])
# plt.show()


#print('matrix_nufft: ', cp.real(pmynufft).get())


#plt.show()

ATA = cp.conjugate(cp.transpose(nufft_matrix, (1,0))) @ nufft_matrix
psf = fourier.toeplitz_psf(coord, shape=(NX,), oversamp=4.0, width=8)

my_ipsf = cp.empty((2*NX,), dtype=cp.complex64)
ipsf_left = cp.squeeze(ATA[:,0])
ipsf_right = cp.flip(cp.squeeze(ATA[0,1:]))

my_ipsf[0:NX] = ipsf_left
#my_ipsf[(NX+1):] = ipsf_right
my_ipsf[(NX+1):] = cp.conjugate(cp.flip(ipsf_left[1:]))
#my_ipsf[(NX+1):] = ipsf_right
#my_ipsf[NX] = (cp.real(my_ipsf[NX-1] + my_ipsf[NX+1])) * 0.5 + 0.5j * (cp.imag(my_ipsf[NX-1]) + cp.imag(my_ipsf[NX+1]))
#my_ipsf[NX] = (my_ipsf[NX-1] + my_ipsf[NX+1]) * 0.5
my_ipsf[NX] = my_ipsf[0] #my_ipsf[NX+1]*0.5 + my_ipsf[NX-1]*0.5
#my_ipsf[NX] = my_ipsf[0]
my_ipsf /= NX

#my_ipsf = cp.fft.fftshift(my_ipsf)

#my_ipsf = cp.fft.ifftshift(my_ipsf)

ipsf = psf
ipsf = cp.fft.ifftshift(ipsf)
ipsf = cp.fft.ifft(ipsf)
#ipsf = cp.fft.fftshift(ipsf)
#ipsf = cp.fft.ifft(psf)

plt.figure()
plt.title('Real ipsf')
plt.plot(cp.real(ipsf).get())
plt.plot(cp.real(my_ipsf).get())
plt.legend(['ipsf', 'my_ipsf'])
plt.show()

plt.figure()
plt.title('Imag ipsf')
plt.plot(cp.imag(ipsf).get())
plt.plot(cp.imag(my_ipsf).get())
plt.legend(['ipsf', 'my_ipsf'])
plt.show()

plt.figure()
plt.title('Abs ipsf')
plt.plot(cp.abs(ipsf).get())
plt.plot(cp.abs(my_ipsf).get())
plt.legend(['ipsf', 'my_ipsf'])
plt.show()

print('Hello')
