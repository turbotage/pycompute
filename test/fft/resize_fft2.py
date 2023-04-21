
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

N1 = 3
N2 = 2
N3 = 2
NX = N1*N2*N3

large = (1,2*N1,2*N2,2*N3)
small = (1, N1, N2, N3)

diag = cp.array((np.random.rand(*large).astype(np.float32) + 1j*np.random.rand(*large).astype(np.float32)).astype(np.complex64))

R = linop.Resize(large, small)

F = fulinops.FFT(large, (1,2,3))

P = linop.Multiply(large, diag)

#A = R.H * F.H
A = R.H * F.H
#A = F.H

p = cp.array((np.random.rand(*large).astype(np.float32) + 1j*np.random.rand(*large).astype(np.float32)).astype(np.complex64))
#p = cp.arange(2*N1*2*N2*2*N3).reshape(1,2*N1,2*N2,2*N3)

out1 = A.apply(p)
out1 = out1.get().flatten()

#out2 = cp.fft.ifftn(p, (N1,N2,N3), (1,2,3))
out2 = p
#out2 = out2[:,:N1,:N2,:N3]
out2 = cp.fft.fftshift(out2, axes=(1,2,3))
out2 = cp.fft.ifftn(out2, axes=(1,2,3), norm="ortho")
out2 = cp.fft.ifftshift(out2, axes=(1,2,3))
out2 = out2[:,:N1,:N2,:N3]
out2 = out2.get().flatten()

plt.figure()
plt.plot(np.abs(out1), 'r-')
plt.plot(np.abs(out2), 'g-')
plt.legend(['nufft1', 'nufft2'])
plt.show()

plt.figure()
plt.plot(np.real(out1), 'r-')
plt.plot(np.real(out2), 'g-')
plt.legend(['nufft1', 'nufft2'])
plt.show()

plt.figure()
plt.plot(np.imag(out1), 'r-')
plt.plot(np.imag(out2), 'g-')
plt.legend(['nufft1', 'nufft2'])
plt.show()