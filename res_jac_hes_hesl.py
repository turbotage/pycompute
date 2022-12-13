import numpy as np
import cupy as cp

import cuda.cuda_program as cuda_cp
from cuda.cuda_program import CudaTensor, CudaFunction
from cuda.symbolic import EvalJacHes
from cuda.solver import GMW81Solver
from cuda.lsqnonlin import ResJacGradHesHesl, SumResGradHesHesl, ResF, GainRatioStep
import math
import time
import torch

ndata = 21
nparam = 4
nconst = 1

batch_size = 1000000

Nelem = batch_size * ndata

pars = CudaTensor([4, batch_size], cp.float32)
pars_t = cuda_cp.from_cu_tensor(pars, rand=True)

consts = CudaTensor([nconst, Nelem], cp.float32)
consts_t = cuda_cp.from_cu_tensor(consts, rand=True)

data = CudaTensor([1, Nelem], cp.float32)
data_t = cuda_cp.from_cu_tensor(data, rand=True)

lam = CudaTensor([1, Nelem], cp.float32)
lam_t = cuda_cp.from_cu_tensor(lam, ones=True)
lam_t = 2*lam_t

step = CudaTensor([4, batch_size], cp.float32)
step_t = cuda_cp.from_cu_tensor(step)

res = CudaTensor([1, Nelem], cp.float32)
res_t = cuda_cp.from_cu_tensor(res)

jac = CudaTensor([nparam, Nelem], cp.float32)
jac_t = cuda_cp.from_cu_tensor(jac)

grad = CudaTensor([nparam, Nelem], cp.float32)
grad_t = cuda_cp.from_cu_tensor(grad)

hes = CudaTensor([round(nparam*(nparam+1)/2), Nelem], cp.float32)
hes_t = cuda_cp.from_cu_tensor(hes)

hesl = CudaTensor([round(nparam*(nparam+1)/2), Nelem], cp.float32)
hesl_t = cuda_cp.from_cu_tensor(hesl)

step_type = CudaTensor([1, batch_size], cp.int8)
step_type_t = cuda_cp.from_cu_tensor(step_type)

fsum = CudaTensor([1, batch_size], cp.float32)
fsum_t = cuda_cp.from_cu_tensor(fsum, zeros=True)

fsum2 = CudaTensor([1, batch_size], cp.float32)
fsum2_t = cuda_cp.from_cu_tensor(fsum2, zeros=True)

gsum = CudaTensor([nparam, batch_size], cp.float32)
gsum_t = cuda_cp.from_cu_tensor(gsum, zeros=True)

hsum = CudaTensor([round(nparam*(nparam+1)/2), batch_size], cp.float32)
hsum_t = cuda_cp.from_cu_tensor(hsum, zeros=True)

hlsum = CudaTensor([round(nparam*(nparam+1)/2), batch_size], cp.float32)
hlsum_t = cuda_cp.from_cu_tensor(hlsum, zeros=True)

expr = 'S0*(f*exp(-b*D_1)+(1-f)*exp(-b*D_2))'
pars_str = ['S0', 'f', 'D_1', 'D_2']
consts_str = ['b']

rjghhlcu = ResJacGradHesHesl(expr, pars_str, consts_str, ndata, cp.float32)
with open("bk_res_jac_grad_hes_hesl.cu", "w") as f:
	f.write(rjghhlcu.build())

summercu = SumResGradHesHesl(nparam, ndata, cp.float32)
with open("bk_summer.cu", "w") as f:
	f.write(summercu.build())

gmw81solcu = GMW81Solver(nparam, cp.float32)
with open("bk_gmw81sol.cu", "w") as f:
	f.write(gmw81solcu.build())

rescu = ResF(expr, pars_str, consts_str, ndata, cp.float32)
with open("bk_res.cu", "w") as f:
	f.write(rescu.build())

gainstepcu = GainRatioStep(nparam, cp.float32)
with open("bk_gain_ratio_step.cu", "w") as f:
	f.write(gainstepcu.build())


Amat = None
bvec = None
Lmat = None
Dmat = None

start = time.time()
for i in range(0,10):
	fsum_t[:,:] = 0.0
	fsum2_t[:,:] = 0.0

	rjghhlcu.run(pars_t, consts_t, data_t, lam_t, res_t, jac_t, grad_t, hes_t, hesl_t)
	summercu.run(res_t, grad_t, hes_t, hesl_t, fsum_t, gsum_t, hsum_t, hlsum_t)
	gmw81solcu.run(hlsum_t, -gsum_t, step_t)
	pars_tp = pars_t + step_t
	rescu.run(pars_tp, consts_t, data_t, res_t, fsum2_t)
	gainstepcu.run(fsum_t, fsum2_t, pars_tp, step_t, gsum_t, hes_t, pars_t, lam_t, step_type_t)
	
	#print('pars: ', pars_t[:,0])
	#print('lam: ', lam_t[:,0])
	#print('step_type: ', step_type_t[:,0])
	#print('eig: ', cp.linalg.eigvalsh(compact_to_full(hlsum_t[:,0])))
	#print('step: ', step_t[:,0])
	#print('step_length: ', cp.linalg.norm(step_t[:,0]))
	#print('End of iter')
	


cp.cuda.stream.get_current_stream().synchronize()
end = time.time()
print('It took: ' + str(end - start) + ' s')

printing1 = False
if printing1:
	print('Solutions: ')
	print(cp.linalg.solve(Amat, bvec))
	print(cp.linalg.solve(Lmat @ Dmat @ Lmat.transpose(), bvec))
	print(step_t[:,0])

	print('LD')
	print(Lmat)
	print(Dmat)

	print('Eigenvalues: ')
	print(cp.linalg.eigvalsh(Amat))
	print(cp.linalg.eigvalsh(Lmat @ Dmat @ Lmat.transpose()))

	print('Matrices: ')
	print(Amat)
	print(Lmat @ Dmat @ Lmat.transpose())
	print(Lmat @ Dmat @ Lmat.transpose() - Amat)


ns = [0, batch_size - 1]
printing2=False
if printing2:
	for ni in ns:
		print('Show Iter: ')
		pt = pars_t[:,ni]
		ct = consts_t[:,ni]
		dt = data_t[:,ni]
		rt = res_t[:,ni]
		gt = gsum[:,ni]
		hlt = hlsum[:,ni]
		ht = hsum[:,ni]
		st = step_t[:,ni]

		print(pt)
		print(ct)
		print(dt)
		print(rt)
		print(gt)
		print(hlt)
		print(ht)
		print(st)


