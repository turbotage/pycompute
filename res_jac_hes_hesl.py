import numpy as np
import cupy as cp

import cuda.cuda_program as cuda_cp
from cuda.cuda_program import CudaTensor, CudaFunction
from cuda.symbolic import EvalJacHes, ResJacGradHesHesl, SumResGradHesHesl, NLSQ_LM
from cuda.solver import GMW81Solver
import math
import time
import torch

ndata = 21
nparam = 4
nconst = 1

batch_size = 1000000

Nelem = batch_size * ndata

def from_cu_tensor(t: CudaTensor, rand=False, zeros=False, ones=False):
	if rand:
		return cp.random.rand(*(t.shape), dtype=t.dtype)
	if zeros:
		return cp.zeros(shape=tuple(t.shape), dtype=t.dtype)
	if ones:
		return cp.ones(shape=tuple(t.shape), dtype=t.dtype)

	return cp.empty(shape=tuple(t.shape), dtype=t.dtype)

pars = CudaTensor([4, batch_size], cp.float32)
pars_t = from_cu_tensor(pars, rand=True)

consts = CudaTensor([nconst, Nelem], cp.float32)
consts_t = from_cu_tensor(consts, rand=True)

data = CudaTensor([1, Nelem], cp.float32)
data_t = from_cu_tensor(data, rand=True)

lam = CudaTensor([1, Nelem], cp.float32)
lam_t = from_cu_tensor(lam, ones=True)
lam_t = 2*lam_t

step = CudaTensor([4, batch_size], cp.float32)
step_t = from_cu_tensor(step)

res = CudaTensor([1, Nelem], cp.float32)
res_t = from_cu_tensor(res)

jac = CudaTensor([nparam, Nelem], cp.float32)
jac_t = from_cu_tensor(jac)

grad = CudaTensor([nparam, Nelem], cp.float32)
grad_t = from_cu_tensor(grad)

hes = CudaTensor([round(nparam*(nparam+1)/2), Nelem], cp.float32)
hes_t = from_cu_tensor(hes)

hesl = CudaTensor([round(nparam*(nparam+1)/2), Nelem], cp.float32)
hesl_t = from_cu_tensor(hesl)

fsum = CudaTensor([1, batch_size], cp.float32)
fsum_t = from_cu_tensor(fsum, zeros=True)

gsum = CudaTensor([nparam, batch_size], cp.float32)
gsum_t = from_cu_tensor(gsum, zeros=True)

hsum = CudaTensor([round(nparam*(nparam+1)/2), batch_size], cp.float32)
hsum_t = from_cu_tensor(hsum, zeros=True)

hlsum = CudaTensor([round(nparam*(nparam+1)/2), batch_size], cp.float32)
hlsum_t = from_cu_tensor(hlsum, zeros=True)

expr = 'S0*(f*exp(-b*D_1)+(1-f)*exp(-b*D_2))'
pars_str = ['S0', 'f', 'D_1', 'D_2']
consts_str = ['b']

rjghhl = ResJacGradHesHesl(expr, pars_str, consts_str, ndata, cp.float32)
rjghhl_code = cuda_cp.code_gen_walking(rjghhl, "")

with open("bk_res_jac_grad_hes_hesl.cu", "w") as f:
	f.write(rjghhl.build())

summer = SumResGradHesHesl(nparam, ndata, cp.float32)
summer_code = cuda_cp.code_gen_walking(summer, "")

with open("bk_summer.cu", "w") as f:
	f.write(summer.build())

gmw81sol = GMW81Solver(nparam, cp.float32)
gmw81sol_code = cuda_cp.code_gen_walking(gmw81sol, "")

with open("bk_gmw81sol.cu", "w") as f:
	f.write(gmw81sol.build())

Amat = None
bvec = None
Lmat = None
Dmat = None

def compact_to_full(mat):
	nmat = mat.shape[0]
	n = math.floor(math.sqrt(2*nmat))
	retmat = cp.empty((n,n))
	k = 0
	for i in range(0,n):
		for j in range(0,i+1):
			retmat[i,j] = mat[k]
			if i != j:
				retmat[j,i] = mat[k]
			k += 1
	return retmat

def compact_to_LD(mat):
	nmat = mat.shape[0]
	n = math.floor(math.sqrt(2*nmat))
	L = cp.zeros((n,n))
	D = cp.zeros((n,n))
	k = 0
	for i in range(0,n):
		for j in range(0,i+1):
			if i != j:
				L[i,j] = mat[k]
			else:
				L[i,j] = 1.0
				D[i,j] = mat[k]
			k += 1
	return (L,D)

start = time.time()
for i in range(0,100):
	rjghhl.run(pars_t, consts_t, data_t, lam_t, res_t, jac_t, grad_t, hes_t, hesl_t, Nelem)
	summer.run(res_t, grad_t, hes_t, hesl_t, fsum_t, gsum_t, hsum_t, hlsum_t, Nelem)
	gmw81sol.run(hlsum_t, -gsum_t, step_t, batch_size)
	pars_t += step_t


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

cp.cuda.stream.get_current_stream().synchronize()



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


