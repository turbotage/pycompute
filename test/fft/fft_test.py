import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from testrun import test_runner
test_runner()

import cupy as cp
import numpy as np

import time
import finufft

from pycompute.cuda.fft import DftT3

from pycompute.cuda import cuda_program as cuda_cp

N1 = 256
N2 = 256
N3 = 256

NX = 2000000
NF = 80000

wx_np = np.random.uniform(-np.pi, np.pi, NF).astype(np.float32)
wy_np = np.random.uniform(-np.pi, np.pi, NF).astype(np.float32)
wz_np = np.random.uniform(-np.pi, np.pi, NF).astype(np.float32)

w_cu = cp.empty((3, NF), dtype=cp.float32)
w_cu[0, :] = cp.array(wx_np)
w_cu[1, :] = cp.array(wy_np)
w_cu[2, :] = cp.array(wz_np)

px_np = np.random.uniform(-N1 / 2, N1 / 2, NX).astype(np.float32)
py_np = np.random.uniform(-N1 / 2, N1 / 2, NX).astype(np.float32)
pz_np = np.random.uniform(-N1 / 2, N1 / 2, NX).astype(np.float32)

p_cu = cp.empty((3, NX), dtype=cp.float32)
p_cu[0, :] = cp.array(px_np)
p_cu[1, :] = cp.array(py_np)
p_cu[2, :] = cp.array(pz_np)

var_x = (np.random.standard_normal((NX,)) + 1j * np.random.standard_normal((NX,))).astype(np.complex64)
var_x_cu = cp.array(var_x)

var_f = np.empty((NF,), dtype=cp.complex64)
var_f_cu = cp.empty_like(var_f)

start = time.time()
finufft.nufft3d3(px_np, py_np, pz_np, var_x, wx_np, wy_np, wz_np, out=var_f, eps=1e-12)
end = time.time()
print(end - start)

dft_obj = DftT3(dtype=cp.float32, sign_positive=True, write_to_file=True)
dft_obj.build()

cp.cuda.get_current_stream().synchronize()
start = time.time()
dft_obj.run(p_cu, w_cu, var_x_cu, var_f_cu)
cp.cuda.get_current_stream().synchronize()
end = time.time()
print(end - start)

RE = np.linalg.norm(var_f_cu.get() - var_f) / np.linalg.norm(var_f)
print(RE)
