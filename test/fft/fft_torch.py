
import time

import torch

fin = torch.rand(512,512,512, dtype=torch.complex64, device=torch.device('cuda:0'))
torch.cuda.synchronize()

start = time.time()

fout = torch.fft.fftn(fin)
for i in range(10):
	fout = torch.fft.fftn(fin + 1)

torch.cuda.synchronize()

end = time.time()
print(end - start)