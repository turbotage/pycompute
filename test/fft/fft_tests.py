

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
t = 100

a = cp.empty((t,N1,N2,N3), dtype=cp.complex64)

cp.cuda.get_current_stream().synchronize()

start = time.time()

f = cp.fft.fft(a, axis=0)
cp.cuda.get_current_stream().synchronize()

end = time.time()

print(end - start)


