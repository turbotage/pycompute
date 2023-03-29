
import time

import numpy as np
import cupy as cp
import cupyx as cpx

fin = (cp.random.uniform(0,1,size=(512,512,512), dtype=cp.float32) + 
	1j * cp.random.uniform(0,1,size=(512,512,512), dtype=cp.float32)).astype(cp.complex64)

cp.cuda.get_current_stream().synchronize()

start = time.time()

fout = cp.fft.fftn(fin, axes=(0,1,2))
for i in range(10):
	fout = cp.fft.fftn(fout + 1)

cp.cuda.get_current_stream().synchronize()

end = time.time()
print(end - start)
