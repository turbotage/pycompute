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


NX = 32
NF = 32

coord = cp.empty((3,NF), dtype=cp.float32)
#coord[0,:] = cp.random.uniform(-NX // 2, NX // 2, NF).astype(cp.float32)
for i in range(NF):
	coord[0,i] = i - NX // 2 #2*cp.pi*i/NF
coord = cp.transpose(coord, (1,0))

psf = fourier.toeplitz_psf(coord, shape=(NX,), oversamp=2.0, width=4)

#my_psf = cp.zeros_like(psf, dtype=cp.complex64)

#first_half = cp.sinc(-cp.arange(NX).astype(cp.float32) / (2*cp.pi))
#second_half = cp.sinc(cp.arange(NX).astype(cp.float32) / (2*cp.pi))

temp_coord = coord
#temp_coord = cp.fft.ifftshift(coord)

my_ipsf = cp.empty((2*NX,), dtype=cp.complex64)
for i in range(NX):
	for j in range(NF):
		my_ipsf[i] += cp.squeeze(cp.exp(-2*cp.pi*1j*temp_coord[j]*i/NX))
for i in range(1,NX):
	for j in range(NF):
		my_ipsf[2*NX-i] += cp.squeeze(cp.exp(2*cp.pi*1j*temp_coord[j]*i/NX))
my_ipsf[NX] = my_ipsf[0]

my_ipsf /= NX

my_psf = cp.fft.fft(my_ipsf)


p = (cp.random.uniform(0,1,(2*NX,)) + 1j*cp.random.uniform(0,1,(2*NX,))).astype(cp.complex64)



ipsf = cp.fft.ifft(psf)

plot_psf = True

if plot_psf:
	plt.figure()
	plt.title('Real')
	plt.plot(cp.real(psf).get())
	plt.plot(cp.real(my_psf).get())
	plt.legend(['psf', 'my_psf'])
	plt.show()

	plt.figure()
	plt.title('Imag')
	plt.plot(cp.imag(psf).get())
	plt.plot(cp.imag(my_psf).get())
	plt.legend(['psf', 'my_psf'])
	plt.show()

	plt.figure()
	plt.title('Abs')
	plt.plot(cp.abs(psf).get())
	plt.plot(cp.abs(my_psf).get())
	plt.legend(['psf', 'my_psf'])
	plt.show()
else:
	plt.figure()
	plt.title('Real')
	plt.plot(cp.real(ipsf).get())
	plt.plot(cp.real(my_ipsf).get())
	plt.legend(['ipsf', 'my_ipsf'])
	plt.show()

	plt.figure()
	plt.title('Imag')
	plt.plot(cp.imag(ipsf).get())
	plt.plot(cp.imag(my_ipsf).get())
	plt.legend(['ipsf', 'my_ipsf'])
	plt.show()

	plt.figure()
	plt.title('Abs')
	plt.plot(cp.abs(ipsf).get())
	plt.plot(cp.abs(my_ipsf).get())
	plt.legend(['ipsf', 'my_ipsf'])
	plt.show()


print('Hello')
