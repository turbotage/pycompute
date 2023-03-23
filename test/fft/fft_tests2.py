


import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from testrun import test_runner
test_runner()

import numpy as np
import cupy as cp
import cupyx as cpx

import pycompute.cuda.sigpy.fourier_linops as fulinops
import pycompute.cuda.sigpy.fourier as fourier
import pycompute.cuda.sigpy.linop as linop

#diag = cp.random.rand(10)
#print(diag)
#diag = cp.array([0.34018924, 0.45034562, 0.68102629, 0.38775271, 0.13584188, 0.99800008,
# 0.1649308, 0.30409642, 0.75138961, 0.04201398])

#diag = cp.array([0.34018924, 0.45034562, 0.68102629, 0.38775271, 0.13584188])
diag = cp.random.rand(512,512,256)

ones = cp.arange(5)

a1 = cp.random.rand(512,512,256)



cp.cuda.get_current_stream().synchronize()

start = time.time()
a2 = fourier.fft(a1, center=True)
a2 = diag * a2
a2 = fourier.ifft(a2, center=True)
cp.cuda.get_current_stream().synchronize()
end = time.time()
print(end - start)


diag_shift = cp.fft.fftshift(diag)
fft_plan = cpx.scipy.fftpack.get_fft_plan(a1)

cp.cuda.get_current_stream().synchronize()

start = time.time()
#b2 = cp.fft.fftn(a1)
b2 = cpx.scipy.fftpack.fftn(a1, plan=fft_plan)
#b3 = alter * b2
b2 = diag_shift * b2
#b3 = alter * b3
#b2 = cp.fft.ifftn(b2)
b2 = cpx.scipy.fftpack.ifftn(b2, plan=fft_plan)
cp.cuda.get_current_stream().synchronize()
end = time.time()
print(end - start)


relerr = cp.linalg.norm(a2 - b2) / cp.linalg.norm(a2)
print(relerr.get())

# N1 = 60
# N2 = 60
# N3 = 60
# NX = N1*N2*N3
# NF = 80000
# t = 10

# x = (cp.random.uniform(0.0,1.0, (t,N1,N2,N3)) + 1j * cp.random.uniform(0.0,1.0, (t,N1,N2,N3))).astype(cp.complex64)
# diag = (cp.random.uniform(0.0,1.0, (t,N1,N2,N3)) + 1j * cp.random.uniform(0.0,1.0, (t,N1,N2,N3))).astype(cp.complex64)

# F1 = fulinops.FFT(x.shape, axes=(1,2,3), center=True)
# F2 = fulinops.FFT(x.shape, axes=(1,2,3), center=False)

# M1 = linop.Multiply(diag.shape, diag)
# M2 = linop.Multiply(diag.shape, diag)

# W1 = F1.H * M1 * F1
# W2 = F2.H * M2 * F2

# b1 = W1 * x
# b2 = W2 * x

# #b2 = cp.fft.fftshift(b2, axes=(1,2,3))

# relerr = np.linalg.norm(b1 - b2) / np.linalg.norm(b1)

# print(relerr.get())