

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

N1 = 60
N2 = 60
N3 = 60
NX = N1*N2*N3
NF = 80000
t = 60

a = (cp.random.uniform(0.0,1.0, (t,N1,N2,N3)) + 1j * cp.random.uniform(0.0,1.0, (t,N1,N2,N3))).astype(cp.complex64)
diag = (cp.random.uniform(0.0,1.0, (t,N1,N2,N3)) + 1j * cp.random.uniform(0.0,1.0, (t,N1,N2,N3))).astype(cp.complex64)


## CuPy only
cp.cuda.get_current_stream().synchronize()

start = time.time()

fcp = cp.fft.fft(a, axis=0)
fcp = diag * fcp
acp = cp.fft.ifft(fcp, axis=0)

cp.cuda.get_current_stream().synchronize()

end = time.time()

print(end - start)

## SigPy
cp.cuda.get_current_stream().synchronize()

start = time.time()

fsp = fourier.fft(a, axes=(0,))
fsp = diag * fsp
asp = fourier.ifft(fsp, axes=(0,))

cp.cuda.get_current_stream().synchronize()

end = time.time()

print(end - start)


relerr = cp.linalg.norm(asp - acp) / cp.linalg.norm(asp)

print(relerr.get())

