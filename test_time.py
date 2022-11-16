import numpy as np
import cupy as cp

import time

nparam = 4
nhes = round(nparam*(nparam+1)/2)
ndata = 21
nbatch = 1000000

hes = cp.empty((nhes, ndata * nbatch))
jac = cp.empty((nparam, ndata * nbatch))

cs = []
start = time.time()
for i in range(0,10):
	hes = cp.empty((nhes, ndata * nbatch))
	jac = cp.empty((nparam, ndata * nbatch))

	hest = hes.transpose().reshape(nbatch, ndata, nhes)
	jact = jac.transpose().reshape(nbatch, ndata, nparam)

	hessum = hest.sum(axis=1)
	hessum += cp.matmul(cp.transpose(jact, (0,2,1)), jact)
	
cp.cuda.stream.get_current_stream().synchronize()
end = time.time()
print('Time: ' + str(end - start))

print(cs)

