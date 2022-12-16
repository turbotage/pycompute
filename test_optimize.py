import cuda.lsqnonlin as clsq

import numpy as np
import cupy as cp
import cuda.cuda_program as cuda_cp

import math
import time

batch_size = 2000000
ndata = 21
Nelem = batch_size * ndata

expr = 'S0*(f*exp(-b*D_1)+(1-f)*exp(-b*D_2))'
pars_str = ['S0', 'f', 'D_1', 'D_2']
consts_str = ['b']

data = np.random.rand(1, Nelem).astype(dtype=cp.float32, copy=True, order='C')
consts = 0.01*np.random.rand(1, Nelem).astype(dtype=cp.float32, copy=True, order='C')
pars = np.random.rand(4, batch_size).astype(dtype=cp.float32, copy=True, order='C')
lower_bound = (np.finfo(np.float32).min / 10.0)*np.ones((4, batch_size), dtype=np.float32)
upper_bound = (np.finfo(np.float32).max / 10.0)*np.ones((4, batch_size), dtype=np.float32)

first_f = np.empty((1,batch_size), dtype=np.float32)
last_f = np.empty((1, batch_size), dtype=np.float32)

nchunks = 1
chunk_size = math.ceil(batch_size / nchunks)

solm = clsq.SecondOrderLevenbergMarquardt(expr, pars_str, consts_str, ndata=21, dtype=cp.float32, write_to_file=True)

start = time.time()
for i in range(0,nchunks):
	
	parscu = cp.array(pars[:,i*chunk_size:(i+1)*chunk_size], dtype=cp.float32, copy=True, order='C')
	constscu = cp.array(consts[:,i*chunk_size*ndata:(i+1)*chunk_size*ndata], dtype=cp.float32, copy=True, order='C')
	datacu = cp.array(data[:,i*chunk_size*ndata:(i+1)*chunk_size*ndata], dtype=cp.float32, copy=True, order='C')
	lower_bound_cu = cp.array(lower_bound[:,i*chunk_size:(i+1)*chunk_size], dtype=cp.float32, copy=True, order='C')
	upper_bound_cu = cp.array(upper_bound[:,i*chunk_size:(i+1)*chunk_size], dtype=cp.float32, copy=True, order='C')

	solm.setup(parscu, constscu, datacu, lower_bound_cu, upper_bound_cu)
	solm.run(20, 1e-30)

	first_f[:,i*chunk_size:(i+1)*chunk_size] = solm.first_f.get()
	last_f[:,i*chunk_size:(i+1)*chunk_size] = solm.last_f.get()

	pars[:,i*chunk_size:(i+1)*chunk_size] = parscu.get()

cp.cuda.stream.get_current_stream().synchronize()
end = time.time()
print('It took: ' + str(end - start) + ' s')

