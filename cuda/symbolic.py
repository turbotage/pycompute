import math
import numpy as np

import cupy as cp
from jinja2 import Template
from symengine import sympify


#from ..sym import sym
from sym import util

from . import linalg
from . import cuda_program as cudap
from .cuda_program import CudaFunction, CudaTensor
from .cuda_program import CudaTensorChecking as ctc


def eval_jac_hes_funcid(expr: str, pars_str: list[str], consts_str: list[str], dtype: cp.dtype):
	return 'eval_jac_hes' + ctc.dim_dim_type_funcid(len(pars_str), len(consts_str), 
		dtype, 'eval_jac_hes') + '_' + util.expr_hash(expr, 12)

def eval_jac_hes_code(expr: str, pars_str: list[str], consts_str: list[str], dtype: cp.dtype):
	
	rjh_temp = Template(
"""
__device__
void {{funcid}}(const {{fp_type}}* params, const {{fp_type}}* consts, {{fp_type}}* eval, 
	{{fp_type}}* jac, {{fp_type}}* hes, int tid, int N) 
{
	{{fp_type}} pars[{{nparam}}];
	for (int i = 0; i < {{nparam}}; ++i) {
		pars[i] = params[i*N+tid];
	}

{{sub_expr}}

{{eval_expr}}

{{jac_expr}}

{{hes_expr}}

}
""")

	type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'eval_jac_hes')

	nparam = len(pars_str)
	nconst = len(consts_str)

	funcid = eval_jac_hes_funcid(expr, pars_str, consts_str, dtype)

	sym_expr = sympify(expr)
	# convert parameter names to ease kernel generation
	for k in range(0,len(pars_str)):
		temp = pars_str[k]
		pars_str[k] = 'parvar_' + temp
		sym_expr = sym_expr.subs(temp, pars_str[k])

	for k in range(0,len(consts_str)):
		temp = consts_str[k]
		consts_str[k] = 'convar_' + temp
		sym_expr = sym_expr.subs(temp, consts_str[k])

	substs, reduced = util.res_jac_hes(str(sym_expr), pars_str, consts_str)
	cuprint = util.CUDAPrinter()

	sub_str = ""
	for s in substs:
		sub_str += '\t'+type+' '+cuprint.tcs_f(s[0])+' = '+cuprint.tcs_f(s[1])+';\n'

	eval_str = '\teval[tid] = '+cuprint.tcs_f(reduced[0])+';'

	jac_str = ""
	for k in range(nparam):
		s = reduced[1+k]
		ctstr = ""
		if dtype == cp.float32:
			ctstr = cuprint.tcs_f(s)
		else:
			ctstr = cuprint.tcs_d(s)
		if ctstr == '0':
			ctstr = '0.0f'
		jac_str += '\tjac['+str(k)+'*N+tid] = '+ctstr+';\n'

	hes_str = ""
	for k in range(round(nparam*(nparam+1)/2)):
		s = reduced[(nparam + 1) + k]
		ctstr = ""
		if dtype == cp.float32:
			ctstr = cuprint.tcs_f(s)
		else:
			ctstr = cuprint.tcs_d(s)
		if ctstr == '0':
			ctstr = '0.0f'
		hes_str += '\thes['+str(k)+'*N+tid] = '+ctstr+';\n'

	for k in range(len(pars_str)):
		p = pars_str[k]
		repl = 'pars['+str(k)+']'
		sub_str = sub_str.replace(p, repl)
		eval_str = eval_str.replace(p, repl)
		jac_str = jac_str.replace(p, repl)
		hes_str = hes_str.replace(p, repl)

	for k in range(len(consts_str)):
		c = consts_str[k]
		repl = 'consts['+str(k)+'*N+tid]'
		sub_str = sub_str.replace(c, repl)
		eval_str = eval_str.replace(c, repl)
		jac_str = jac_str.replace(c, repl)
		hes_str = hes_str.replace(c, repl)

	rjh_kernel = rjh_temp.render(funcid=funcid, fp_type=type,
		nparam=nparam, nconst=nconst,
		sub_expr=sub_str, eval_expr=eval_str, jac_expr=jac_str, hes_expr=hes_str)


	return rjh_kernel

class EvalJacHes(CudaFunction):
	def __init__(self, expr: str, pars_str: list[str], consts_str: list[str], dtype: cp.dtype):

		self.type_str = ctc.type_to_typestr(dtype)
		self.pars_str = pars_str
		self.consts_str = consts_str

		self.funcid = eval_jac_hes_funcid(expr, pars_str, consts_str, dtype)
		self.code = eval_jac_hes_code(expr, pars_str, consts_str, dtype)

		self.mod = None
		self.run_func = None

	def run(self, pars, consts, eval, jac, hes, N):
		if self.run_func == None:
			self.build()

		Nthreads = 32
		blockSize = np.ceil(N / Nthreads)
		self.run_func((blockSize,),(Nthreads,),(pars, consts, eval, jac, hes, N))

	def get_device_funcid(self):
		return self.funcid

	def get_kernel_funcid(self):
		funcid = self.funcid
		return 'k_' + funcid

	def get_device_code(self):
		return self.code

	def get_kernel_code(self):
		temp = Template(
"""
extern \"C\" __global__
void {{funcid}}(const {{fp_type}}* pars, const {{fp_type}}* consts,
	{{fp_type}}* eval, {{fp_type}}* jac, {{fp_type}}* hes, int N) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N) {
		{{dfuncid}}(pars, consts, eval, jac, hes, tid, N);
	}
}
""")
		fid = self.get_kernel_funcid()
		dfid = self.get_device_funcid()

		return temp.render(funcid=fid, dfuncid=dfid, fp_type=self.type_str)

	def get_full_code(self):
		code = self.get_device_code() + '\n'
		code += self.get_kernel_code()
		return code

	def build(self):
		cc = cudap.code_gen_walking(self, "")
		self.mod = cp.RawModule(code=cc)
		self.run_func = self.mod.get_function(self.get_kernel_funcid())
		return cc

	def get_deps(self):
		return list()



def res_jac_grad_hes_hesl_funcid(expr: str, pars_str: list[str], consts_str: list[str], dtype: cp.dtype):
	return 'res_jac_grad_hes_hesl' + ctc.dim_dim_type_funcid(len(pars_str), len(consts_str), 
		dtype, 'res_jac_grad_hes_hesl') + '_' + util.expr_hash(expr, 12)

def res_jac_grad_hes_hesl_code(expr: str, pars_str: list[str], consts_str: list[str], dtype: cp.dtype):
	
	rjh_temp = Template(
"""
__device__
void {{funcid}}(const {{fp_type}}* params, const {{fp_type}}* consts, const {{fp_type}}* data, const {{fp_type}}* lam,
	{{fp_type}}* res, {{fp_type}}* jac, {{fp_type}}* grad, {{fp_type}}* hes, {{fp_type}}* hesl, int tid, int N) 
{
	{{fp_type}} pars[{{nparam}}];
	#pragma unroll
	for (int i = 0; i < {{nparam}}; ++i) {
		pars[i] = params[i*N+tid];
	}

{{sub_expr}}

{{eval_expr}}

{{jac_expr}}

{{hes_expr}}
		
	int k = 0;
	#pragma unroll
	for (int i = 0; i < {{nparam}}; ++i) {
		for (int j = 0; j <= i; ++j) {
			float jtemp = jac[i*N+tid] * jac[j*N+tid];
			hes[k*N+tid] += jtemp;
			if (i != j) {
				hesl[k*N+tid] = hes[k*N+tid];
			} else {
				hesl[k*N+tid] = hes[k*N+tid] + lam[tid]*jtemp;
			}
			++k;
		}
	}

	#pragma unroll
	for (int i = 0; i < {{nparam}}; ++i) {
		grad[i*N+tid] = jac[i*N+tid] * res[tid];
	}
}
""")

	type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'res_jac_grad_hes_hesl')

	nparam = len(pars_str)
	nconst = len(consts_str)

	funcid = res_jac_grad_hes_hesl_funcid(expr, pars_str, consts_str, dtype)

	sym_expr = sympify(expr)
	# convert parameter names to ease kernel generation
	for k in range(0,len(pars_str)):
		temp = pars_str[k]
		pars_str[k] = 'parvar_' + temp
		sym_expr = sym_expr.subs(temp, pars_str[k])

	for k in range(0,len(consts_str)):
		temp = consts_str[k]
		consts_str[k] = 'convar_' + temp
		sym_expr = sym_expr.subs(temp, consts_str[k])

	substs, reduced = util.res_jac_hes(str(sym_expr), pars_str, consts_str)
	cuprint = util.CUDAPrinter()

	sub_str = ""
	for s in substs:
		sub_str += '\t'+type+' '+cuprint.tcs_f(s[0])+' = '+cuprint.tcs_f(s[1])+';\n'

	eval_str = '\tres[tid] = '+cuprint.tcs_f(reduced[0])+'-data[tid];'

	jac_str = ""
	for k in range(nparam):
		s = reduced[1+k]
		ctstr = ""
		if dtype == cp.float32:
			ctstr = cuprint.tcs_f(s)
		else:
			ctstr = cuprint.tcs_d(s)
		if ctstr == '0':
			ctstr = '0.0f'
		jac_str += '\tjac['+str(k)+'*N+tid] = '+ctstr+';\n'

	hes_str = ""
	for k in range(round(nparam*(nparam+1)/2)):
		s = reduced[(nparam + 1) + k]
		ctstr = ""
		if dtype == cp.float32:
			ctstr = cuprint.tcs_f(s)
		else:
			ctstr = cuprint.tcs_d(s)
		if ctstr == '0':
			ctstr = '0.0f'
		hes_str += '\thes['+str(k)+'*N+tid] = '+ctstr+'*res[tid];\n'

	for k in range(len(pars_str)):
		p = pars_str[k]
		repl = 'pars['+str(k)+']'
		sub_str = sub_str.replace(p, repl)
		eval_str = eval_str.replace(p, repl)
		jac_str = jac_str.replace(p, repl)
		hes_str = hes_str.replace(p, repl)

	for k in range(len(consts_str)):
		c = consts_str[k]
		repl = 'consts['+str(k)+'*N+tid]'
		sub_str = sub_str.replace(c, repl)
		eval_str = eval_str.replace(c, repl)
		jac_str = jac_str.replace(c, repl)
		hes_str = hes_str.replace(c, repl)

	rjh_kernel = rjh_temp.render(funcid=funcid, fp_type=type,
		nparam=nparam, nconst=nconst,
		sub_expr=sub_str, eval_expr=eval_str, jac_expr=jac_str, hes_expr=hes_str)


	return rjh_kernel

class ResJacGradHesHesl(CudaFunction):
	def __init__(self, expr: str, pars_str: list[str], consts_str: list[str], dtype: cp.dtype):

		self.type_str = ctc.type_to_typestr(dtype)
		self.pars_str = pars_str
		self.consts_str = consts_str

		self.funcid = res_jac_grad_hes_hesl_funcid(expr, pars_str, consts_str, dtype)
		self.code = res_jac_grad_hes_hesl_code(expr, pars_str, consts_str, dtype)

		self.mod = None
		self.run_func = None

	def run(self, pars, consts, data, lam, res, jac, grad, hes, hesl, N):
		if self.run_func == None:
			self.build()

		Nthreads = 32
		blockSize = math.ceil(N / Nthreads)
		self.run_func((blockSize,),(Nthreads,),(pars, consts, data, lam, res, jac, grad, hes, hesl, N))

	def get_device_funcid(self):
		return self.funcid

	def get_kernel_funcid(self):
		funcid = self.funcid
		return 'k_' + funcid

	def get_device_code(self):
		return self.code

	def get_kernel_code(self):
		temp = Template(
"""
extern \"C\" __global__
void {{funcid}}(const {{fp_type}}* params, const {{fp_type}}* consts, const {{fp_type}}* data, const {{fp_type}}* lam,
	{{fp_type}}* res, {{fp_type}}* jac, {{fp_type}}* grad, {{fp_type}}* hes, {{fp_type}}* hesl, int N) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N) {
		{{dfuncid}}(params, consts, data, lam, res, jac, grad, hes, hesl, tid, N);
	}
}
""")
		fid = self.get_kernel_funcid()
		dfid = self.get_device_funcid()

		return temp.render(funcid=fid, dfuncid=dfid, fp_type=self.type_str)

	def get_full_code(self):
		code = self.get_device_code() + '\n'
		code += self.get_kernel_code()
		return code

	def build(self):
		cc = cudap.code_gen_walking(self, "")
		self.mod = cp.RawModule(code=cc)
		self.run_func = self.mod.get_function(self.get_kernel_funcid())
		return cc

	def get_deps(self):
		return list()



def res_hes_funcid(nparam: int, dtype: cp.dtype):
	return 'res_hes' + ctc.dim_type_funcid(nparam, dtype)

def res_hes_code(nparam: int, dtype: cp.dtype):
	
	rjh_temp = Template(
"""
__device__
void {{funcid}}(const {{fp_type}}* data, {{fp_type}}* eval, {{fp_type}}* hes, int tid, int N) 
{
	eval[tid] -= data[tid];

	for (int i = 0; i < {{nhes}}; ++i) {
		hes[i*N+tid] = hes[i*N+tid] * eval[tid];
	}
}
""")

	type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'res_hes')
	funcid = res_hes_funcid(nparam, dtype)
	nhes = round(nparam*(nparam+1)/2)

	rjh_kernel = rjh_temp.render(funcid=funcid, fp_type=type, nhes=nhes)
	return rjh_kernel

class ResHes(CudaFunction):
	def __init__(self, nparam: int, dtype: cp.dtype):
		
		self.funcid = res_hes_funcid(nparam, dtype)
		self.code = res_hes_code(nparam, dtype)

		self.type_str = ctc.type_to_typestr(dtype)

		self.mod = None
		self.run_func = None

	def run(self, pars, consts, eval, jac, hes, N):
		if self.run_func == None:
			self.build()
		Nthreads = 32
		blockSize = np.ceil(N / Nthreads)
		self.run_func((blockSize,),(Nthreads,),(pars, consts, eval, jac, hes, N))

	def get_device_funcid(self):
		return self.funcid

	def get_kernel_funcid(self):
		funcid = self.funcid
		return 'k_' + funcid

	def get_device_code(self):
		return self.code

	def get_kernel_code(self):
		temp = Template(
"""
extern \"C\" __global__
void {{funcid}}(const {{fp_type}}* data, {{fp_type}}* eval, {{fp_type}}* hes, int N) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N) {
		{{dfuncid}}(data, eval, hes, tid, N);
	}
}
""")
		fid = self.get_kernel_funcid()
		dfid = self.get_device_funcid()

		return temp.render(funcid=fid, dfuncid=dfid)

	def get_full_code(self):
		code = self.get_device_code() + '\n'
		code += self.get_kernel_code()
		return code

	def build(self):
		cc = cudap.code_gen_walking(self, "")
		self.mod = cp.RawModule(code=cc)
		self.run_func = self.mod.get_function(self.get_kernel_funcid())
		return cc

	def get_deps(self):
		return list()



class NLSQ_LM:
	
	@staticmethod
	def compact_rjghhl(ndata: int, grad, hes, hesl):
		N = round(grad.shape[1] / ndata)
		nparam = grad.shape[0]
		nhes = round(nparam*(nparam+1)/2)
		hsum = hes.transpose().reshape(N,ndata,nhes).sum(axis=1).transpose()
		hlsum = hesl.transpose().reshape(N,ndata,nhes).sum(axis=1).transpose()
		gsum = grad.transpose().reshape(N,ndata,nparam).sum(axis=1).transpose()
		return (cp.ascontiguousarray(hsum), cp.ascontiguousarray(hlsum), cp.ascontiguousarray(gsum))

	

