import cuda.lsqnonlin as clsq

import numpy as np
import cupy as cp
import cuda.cuda_program as cuda_cp

import math
import time

batch_size = 10000000
ndata = 21
Nelem = batch_size * ndata

expr = 'S0*(f*exp(-b*D_1)+(1-f)*exp(-b*D_2))'
pars_str = ['S0', 'f', 'D_1', 'D_2']
consts_str = ['b']

data = np.random.rand(1, Nelem).astype(dtype=cp.float32)
consts = np.random.rand(1, Nelem).astype(dtype=cp.float32)
pars = np.random.rand(4, batch_size).astype(dtype=cp.float32)
lower_bound = np.random.rand(4, batch_size).astype(dtype=cp.float32)
upper_bound = np.random.rand(4, batch_size).astype(dtype=cp.float32)

nchunks = 10
chunk_size = math.ceil(batch_size / nchunks)

parscu = cp.asarray(pars[:,0:chunk_size].copy())
constscu = cp.asarray(data[:,0:chunk_size*ndata].copy())
datacu = cp.asarray(data[:,0:chunk_size*ndata].copy())
lower_bound_cu = cp.asarray(lower_bound[:,0:chunk_size].copy())
upper_bound_cu = cp.asarray(upper_bound[:,0:chunk_size].copy())

start = time.time()
for i in range(0,nchunks-1):
    
    parscu = cp.asarray(pars[:,i*chunk_size:(i+1)*chunk_size].copy())
    constscu = cp.asarray(consts[:,i*chunk_size*ndata:(i+1)*chunk_size*ndata].copy())
    datacu = cp.asarray(data[:,i*chunk_size*ndata:(i+1)*chunk_size*ndata].copy())
    lower_bound_cu = cp.asarray(lower_bound[:,i*chunk_size:(i+1)*chunk_size].copy())
    upper_bound_cu = cp.asarray(upper_bound[:,i*chunk_size:(i+1)*chunk_size].copy())

    solm = clsq.SecondOrderLevenbergMarquardt(expr, pars_str, consts_str, parscu, constscu, datacu, lower_bound_cu, upper_bound_cu)
    solm.run(50, 1e-5)

    pars[:,i*chunk_size:(i+1)*chunk_size] = parscu.get()

cp.cuda.stream.get_current_stream().synchronize()
end = time.time()
print('It took: ' + str(end - start) + ' s')

