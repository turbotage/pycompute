import cuda.lsqnonlin as clsq

import numpy as np
import cupy as cp
import cuda.cuda_program as cuda_cp

import math
import time

batch_size = 20000
ndata = 21

expr = 'S0*(f*exp(-b*D_1)+(1-f)*exp(-b*D_2))'
pars_str = ['S0', 'f', 'D_1', 'D_2']
consts_str = ['b']

data = np.random.rand(ndata, batch_size).astype(dtype=cp.float32, copy=True, order='C')
consts = 0.01*np.random.rand(1, ndata, batch_size).astype(dtype=cp.float32, copy=True, order='C')
pars = np.random.rand(4, batch_size).astype(dtype=cp.float32, copy=True, order='C')
lower_bound = (np.finfo(np.float32).min / 10.0)*np.ones((4, batch_size), dtype=np.float32)
upper_bound = (np.finfo(np.float32).max / 10.0)*np.ones((4, batch_size), dtype=np.float32)

data[:,0] = np.array([908.02686, 905.39154, 906.08997, 700.7829, 753.0848, 859.9136,
  	   870.48846, 755.96893, 617.3499, 566.2044 , 746.62067, 662.47424,
  	   628.8806, 459.7746 , 643.30554, 318.58453, 416.5493, 348.34335,
  	   411.74026, 284.17468, 290.30487], dtype=np.float32)

with open('bvals_ivim.npy', 'rb') as f:
    consts[:,:,0] = np.load(f).astype(cp.float32).reshape((1,ndata))

pars[:,0] = np.array([700.0, 0.2, 0.1, 0.001], dtype=np.float32)


first_f = np.empty((1,batch_size), dtype=np.float32)
last_f = np.empty((1, batch_size), dtype=np.float32)

nchunks = 1
chunk_size = math.ceil(batch_size / nchunks)

solm = clsq.SecondOrderLevenbergMarquardt(expr, pars_str, consts_str, ndata=21, dtype=cp.float32, write_to_file=True)

start = time.time()

parscu = cp.array(pars, dtype=cp.float32, copy=True, order='C')
constscu = cp.array(consts, dtype=cp.float32, copy=True, order='C')
datacu = cp.array(data, dtype=cp.float32, copy=True, order='C')
lower_bound_cu = cp.array(lower_bound, dtype=cp.float32, copy=True, order='C')
upper_bound_cu = cp.array(upper_bound, dtype=cp.float32, copy=True, order='C')

solm.setup(parscu, constscu, datacu, lower_bound_cu, upper_bound_cu)
solm.run(20, 1e-30)

print(solm.h_t[:,0])

first_f = solm.first_f.get()
last_f = solm.last_f.get()

pars = parscu.get()

cp.cuda.stream.get_current_stream().synchronize()
end = time.time()
print('It took: ' + str(end - start) + ' s')

