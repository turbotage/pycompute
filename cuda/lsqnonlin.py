import math
import numpy as np

import cupy as cp
from jinja2 import Template
from symengine import sympify


#from ..sym import sym
from sym import util

from . import linalg
from . import solver

from . import cuda_program as cudap
from .cuda_program import CudaFunction, CudaTensor
from .cuda_program import CudaTensorChecking as ctc



def resf_funcid(expr: str, pars_str: list[str], consts_str: list[str], ndata: int, dtype: cp.dtype):
	return 'resf' + ctc.dim_dim_dim_type_funcid(len(pars_str), len(consts_str), ndata,
		dtype, 'resf') + '_' + util.expr_hash(expr, 12)

def resf_code(expr: str, pars_str: list[str], consts_str: list[str], ndata: int, dtype: cp.dtype):
	rjh_temp = Template(
"""
__device__
void {{funcid}}(const {{fp_type}}* params, const {{fp_type}}* consts, const {{fp_type}}* data, const char* step_type,
	{{fp_type}}* res, {{fp_type}}* f, int tid, int N, int Nelem) 
{
	{{fp_type}} pars[{{nparam}}];
	int bucket = tid / {{ndata}};
	if (step_type[bucket] == 0) {
		return;
	}

	for (int i = 0; i < {{nparam}}; ++i) {
		pars[i] = params[i*N+bucket];
	}

{{sub_expr}}

{{res_expr}}

	res[tid] = rtid;
	atomicAdd(&f[bucket], rtid*rtid);
}
""")

	type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'res')

	nparam = len(pars_str)
	nconst = len(consts_str)

	funcid = resf_funcid(expr, pars_str, consts_str, ndata, dtype)

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

	substs, reduced = util.res(str(sym_expr), pars_str, consts_str)
	cuprint = util.CUDAPrinter()

	sub_str = ""
	for s in substs:
		sub_str += '\t'+type+' '+cuprint.tcs_f(s[0])+' = '+cuprint.tcs_f(s[1])+';\n'

	res_str = '\tfloat rtid = '+cuprint.tcs_f(reduced[0])+'-data[tid];'

	for k in range(len(pars_str)):
		p = pars_str[k]
		repl = 'pars['+str(k)+']'
		sub_str = sub_str.replace(p, repl)
		res_str = res_str.replace(p, repl)

	for k in range(len(consts_str)):
		c = consts_str[k]
		repl = 'consts['+str(k)+'*Nelem+tid]'
		sub_str = sub_str.replace(c, repl)
		res_str = res_str.replace(c, repl)

	rjh_kernel = rjh_temp.render(funcid=funcid, fp_type=type,
		nparam=nparam, nconst=nconst, ndata=ndata,
		sub_expr=sub_str, res_expr=res_str)


	return rjh_kernel

class ResF(CudaFunction):
	def __init__(self, expr: str, pars_str: list[str], consts_str: list[str], ndata: int, dtype: cp.dtype, write_to_file: bool = False):

		self.type_str = ctc.type_to_typestr(dtype)
		self.pars_str = pars_str.copy()
		self.consts_str = consts_str.copy()
		self.ndata = ndata

		self.write_to_file = write_to_file

		self.funcid = resf_funcid(expr, self.pars_str, self.consts_str, ndata, dtype)
		self.code = resf_code(expr, self.pars_str, self.consts_str, ndata, dtype)

		self.mod = None
		self.run_func = None

	def run(self, pars, consts, data, step_type, res, ftp):
		if self.run_func == None:
			self.build()

		Nelem = consts.shape[1]
		N = round(Nelem / self.ndata)

		Nthreads = 32
		blockSize = math.ceil(Nelem / Nthreads)
		self.run_func((blockSize,),(Nthreads,),(pars, consts, data, step_type, res, ftp, N, Nelem))

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
void {{funcid}}(const {{fp_type}}* params, const {{fp_type}}* consts, const {{fp_type}}* data, const char* step_type,
	{{fp_type}}* res, {{fp_type}}* f, int N, int Nelem) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N) {
		{{dfuncid}}(params, consts, data, step_type, res, f, tid, N, Nelem);
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
		if self.write_to_file:
			with open(self.get_device_funcid() + '.cu', "w") as f:
				f.write(cc)
		try:
			self.mod = cp.RawModule(code=cc)
			self.run_func = self.mod.get_function(self.get_kernel_funcid())
		except:
			with open("on_compile_fail.cu", "w") as f:
				f.write(cc)
			raise
		return cc

	def get_deps(self):
		return list()

def fghhl_code(expr: str, pars_str: list[str], consts_str: list[str], ndata: int, dtype: cp.dtype):
	return 'fghhl' + ctc.dim_dim_dim_type_funcid(len(pars_str), len(consts_str), ndata,
		dtype, 'fghhl') + '_' + util.expr_hash(expr, 12)

def fghhl_code(expr: str, pars_str: list[str], consts_str: list[str], ndata: int, dtype: cp.dtype):
	
	rjh_temp = Template(
"""
__device__
void {{funcid}}(const {{fp_type}}* params, const {{fp_type}}* consts, const {{fp_type}}* data, const {{fp_type}}* lam, const char* step_type,
	{{fp_type}}* f, {{fp_type}}* g, {{fp_type}}* h, {{fp_type}}* hl, int tid, int N, int Nelem) 
{
	int bucket = tid / {{ndata}};
	if (step_type[bucket] == 0) {
		return;
	}

	{{fp_type}} pars[{{nparam}}];
	{{fp_type}} res;

	{{fp_type}} jac[{{nparam}}];
	{{fp_type}} hes[{{nhes}}];
	{{fp_type}} hesl[{{nhes}}];
	{{fp_type}} lambda = lam[tid];


	for (int i = 0; i < {{nparam}}; ++i) {
		pars[i] = params[i*N+bucket];
	}

{{sub_expr}}

{{eval_expr}}

{{jac_expr}}

{{hes_expr}}
		
	int k = 0;
	for (int i = 0; i < {{nparam}}; ++i) {
		for (int j = 0; j <= i; ++j) {
			{{fp_type}} jtemp = jac[i] * jac[j];
			hes[k] += jtemp;
			if (i != j) {
				hesl[k] = hes[k];
			} else {
				hesl[k] = hes[k] + lambda*jtemp;
			}
			++k;
		}
	}

	for (int i = 0; i < {{nparam}}; ++i) {
		int iidx = i*Nelem+tid;
		grad[i] = jac[i] * res;
	}

	res *= res;

	// Start summing up parts
	atomicAdd(&f[bucket], res);

	for (int i = 0; i < {{nparam}}; ++i) {
		atomicAdd(&g[i*N+bucket], grad[i]);
	}

	for (int i = 0; i < {{nhes}}; ++i) {
		int iidx = i*N+bucket;
		atomicAdd(&h[iidx], hes[i]);
		atomicAdd(&hl[iidx], hesl[i]);
	}

}
""")

	type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'fghhl')

	nparam = len(pars_str)
	nconst = len(consts_str)
	nhes = round(nparam*(nparam+1)/2)

	funcid = res_jac_grad_hes_hesl_funcid(expr, pars_str, consts_str, ndata, dtype)

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

	eval_str = '\tres = '+cuprint.tcs_f(reduced[0])+'-data[tid];'

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
		jac_str += '\tjac['+str(k)+'] = '+ctstr+';\n'

	hes_str = ""
	for k in range(nhes)):
		s = reduced[(nparam + 1) + k]
		ctstr = ""
		if dtype == cp.float32:
			ctstr = cuprint.tcs_f(s)
		else:
			ctstr = cuprint.tcs_d(s)
		if ctstr == '0':
			ctstr = '0.0f'
		hes_str += '\thes['+str(k)+'] = '+ctstr+'*res;\n'

	for k in range(nparam):
		p = pars_str[k]
		repl = 'pars['+str(k)+']'
		sub_str = sub_str.replace(p, repl)
		eval_str = eval_str.replace(p, repl)
		jac_str = jac_str.replace(p, repl)
		hes_str = hes_str.replace(p, repl)

	for k in range(len(consts_str)):
		c = consts_str[k]
		repl = 'consts['+str(k)+'*Nelem+tid]'
		sub_str = sub_str.replace(c, repl)
		eval_str = eval_str.replace(c, repl)
		jac_str = jac_str.replace(c, repl)
		hes_str = hes_str.replace(c, repl)

	rjh_kernel = rjh_temp.render(funcid=funcid, fp_type=type,
		nparam=nparam, nconst=nconst, ndata=ndata, nhes=nhes,
		sub_expr=sub_str, eval_expr=eval_str, jac_expr=jac_str, hes_expr=hes_str)


	return rjh_kernel



def res_jac_grad_hes_hesl_funcid(expr: str, pars_str: list[str], consts_str: list[str], ndata: int, dtype: cp.dtype):
	return 'res_jac_grad_hes_hesl' + ctc.dim_dim_dim_type_funcid(len(pars_str), len(consts_str), ndata, 
		dtype, 'res_jac_grad_hes_hesl') + '_' + util.expr_hash(expr, 12)

def res_jac_grad_hes_hesl_code(expr: str, pars_str: list[str], consts_str: list[str], ndata: int, dtype: cp.dtype):
	
	rjh_temp = Template(
"""
__device__
void {{funcid}}(const {{fp_type}}* params, const {{fp_type}}* consts, const {{fp_type}}* data, const {{fp_type}}* lam, const char* step_type,
	{{fp_type}}* res, {{fp_type}}* jac, {{fp_type}}* grad, {{fp_type}}* hes, {{fp_type}}* hesl, int tid, int N, int Nelem) 
{

	{{fp_type}} pars[{{nparam}}];
	int bucket = tid / {{ndata}};

	if (step_type[bucket] == 0) {
		return;
	}

	for (int i = 0; i < {{nparam}}; ++i) {
		pars[i] = params[i*N+bucket];
	}

{{sub_expr}}

{{eval_expr}}

{{jac_expr}}

{{hes_expr}}
		
	int k = 0;
	for (int i = 0; i < {{nparam}}; ++i) {
		for (int j = 0; j <= i; ++j) {
			{{fp_type}} jtemp = jac[i*Nelem+tid] * jac[j*Nelem+tid];
			int kidx = k*Nelem+tid;
			hes[kidx] += jtemp;
			if (i != j) {
				hesl[kidx] = hes[kidx];
			} else {
				hesl[kidx] = hes[kidx] + lam[bucket]*jtemp;
			}
			++k;
		}
	}

	for (int i = 0; i < {{nparam}}; ++i) {
		int iidx = i*Nelem+tid;
		grad[iidx] = jac[iidx] * res[tid];
	}
}
""")

	type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'res_jac_grad_hes_hesl')

	nparam = len(pars_str)
	nconst = len(consts_str)

	funcid = res_jac_grad_hes_hesl_funcid(expr, pars_str, consts_str, ndata, dtype)

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
		jac_str += '\tjac['+str(k)+'*Nelem+tid] = '+ctstr+';\n'

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
		hes_str += '\thes['+str(k)+'*Nelem+tid] = '+ctstr+'*res[tid];\n'

	for k in range(len(pars_str)):
		p = pars_str[k]
		repl = 'pars['+str(k)+']'
		sub_str = sub_str.replace(p, repl)
		eval_str = eval_str.replace(p, repl)
		jac_str = jac_str.replace(p, repl)
		hes_str = hes_str.replace(p, repl)

	for k in range(len(consts_str)):
		c = consts_str[k]
		repl = 'consts['+str(k)+'*Nelem+tid]'
		sub_str = sub_str.replace(c, repl)
		eval_str = eval_str.replace(c, repl)
		jac_str = jac_str.replace(c, repl)
		hes_str = hes_str.replace(c, repl)

	rjh_kernel = rjh_temp.render(funcid=funcid, fp_type=type,
		nparam=nparam, nconst=nconst, ndata=ndata,
		sub_expr=sub_str, eval_expr=eval_str, jac_expr=jac_str, hes_expr=hes_str)


	return rjh_kernel

class ResJacGradHesHesl(CudaFunction):
	def __init__(self, expr: str, pars_str: list[str], consts_str: list[str], ndata: int, dtype: cp.dtype, write_to_file: bool = False):

		self.type_str = ctc.type_to_typestr(dtype)
		self.pars_str = pars_str.copy()
		self.consts_str = consts_str.copy()
		self.ndata = ndata

		self.write_to_file = write_to_file

		self.funcid = res_jac_grad_hes_hesl_funcid(expr, self.pars_str, self.consts_str, ndata, dtype)
		self.code = res_jac_grad_hes_hesl_code(expr, self.pars_str, self.consts_str, ndata, dtype)

		self.mod = None
		self.run_func = None

	def run(self, pars, consts, data, lam, step_type, res, jac, grad, hes, hesl):
		if self.run_func == None:
			self.build()

		Nelem = hes.shape[1]
		N = round(Nelem / self.ndata)

		Nthreads = 32
		blockSize = math.ceil(Nelem / Nthreads)
		self.run_func((blockSize,),(Nthreads,),(pars, consts, data, lam, step_type, res, jac, grad, hes, hesl, N, Nelem))

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
void {{funcid}}(const {{fp_type}}* params, const {{fp_type}}* consts, const {{fp_type}}* data, const {{fp_type}}* lam, const char* step_type,
	{{fp_type}}* res, {{fp_type}}* jac, {{fp_type}}* grad, {{fp_type}}* hes, {{fp_type}}* hesl, int N, int Nelem) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < Nelem) {
		{{dfuncid}}(params, consts, data, lam, step_type, res, jac, grad, hes, hesl, tid, N, Nelem);
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
		if self.write_to_file:
			with open(self.get_device_funcid() + '.cu', "w") as f:
				f.write(cc)
		try:
			self.mod = cp.RawModule(code=cc)
			self.run_func = self.mod.get_function(self.get_kernel_funcid())
		except:
			with open("on_compile_fail.cu", "w") as f:
				f.write(cc)
			raise
		return cc

	def get_deps(self):
		return list()


def sum_res_grad_hes_hesl_funcid(nparam: int, ndata: int, dtype: cp.dtype):
	return 'sum_res_jac_hes_hesl' + ctc.dim_dim_type_funcid(nparam, ndata, dtype)

def sum_res_grad_hes_hesl_code(nparam: int, ndata: int, dtype: cp.dtype):
	temp = Template(
"""
__device__
void {{funcid}}(const {{fp_type}}* res, const {{fp_type}}* grad, const {{fp_type}}* hes, const {{fp_type}}* hesl, const char* step_type,
	{{fp_type}}* f, {{fp_type}}* g, {{fp_type}}* h, {{fp_type}}* hl, int tid, int N, int Nelem) 
{

	int bucket = tid / {{ndata}};
	if (step_type[bucket] == 0) {
		return;
	}

	float rtid = res[tid];
	atomicAdd(&f[bucket], rtid*rtid);
	for (int i = 0; i < {{nparam}}; ++i) {
		atomicAdd(&g[i*N + bucket], grad[i*Nelem+tid]);
	}
	for (int i = 0; i < {{nhes}}; ++i) {
		int i_nb = i*N + bucket;
		int i_nt = i*Nelem + tid;
		atomicAdd(&h[i_nb], hes[i_nt]);
		atomicAdd(&hl[i_nb], hesl[i_nt]);
	}
}
""")

	type = ctc.type_to_typestr(dtype)
	funcid = sum_res_grad_hes_hesl_funcid(nparam, ndata, dtype)
	nhes = round(nparam*(nparam+1)/2)
	return temp.render(funcid=funcid, fp_type=type, nparam=nparam, nhes=nhes, ndata=ndata)

class SumResGradHesHesl(CudaFunction):
	def __init__(self, nparam: int, ndata: int, dtype: cp.dtype, write_to_file: bool = False):
		
		self.funcid = sum_res_grad_hes_hesl_funcid(nparam, ndata, dtype)
		self.code = sum_res_grad_hes_hesl_code(nparam, ndata, dtype)

		self.write_to_file = write_to_file

		self.type_str = ctc.type_to_typestr(dtype)
		self.nparam = nparam
		self.ndata = ndata
		self.dtype = dtype

		self.mod = None
		self.run_func = None

	def run(self, res, grad, hes, hesl, step_type, f, g, h, hl):
		if self.run_func == None:
			self.build()

		Nelem = hes.shape[1]
		N = round(Nelem / self.ndata)
		Nthreads = 32
		blockSize = math.ceil(Nelem / Nthreads)
		self.run_func((blockSize,),(Nthreads,),(res, grad, hes, hesl, step_type, f, g, h, hl, N, Nelem))

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
void {{funcid}}(const {{fp_type}}* res, const {{fp_type}}* grad, const {{fp_type}}* hes, const {{fp_type}}* hesl, const char* step_type,
	{{fp_type}}* f, {{fp_type}}* g, {{fp_type}}* h, {{fp_type}}* hl, int N, int Nelem) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < Nelem) {
		{{dfuncid}}(res, grad, hes, hesl, step_type, f, g, h, hl, tid, N, Nelem);
	}
}
""")
		fid = self.get_kernel_funcid()
		dfid = self.get_device_funcid()

		return temp.render(funcid=fid, fp_type=self.type_str, dfuncid=dfid)

	def get_full_code(self):
		code = self.get_device_code() + '\n'
		code += self.get_kernel_code()
		return code

	def build(self):
		cc = cudap.code_gen_walking(self, "")
		if self.write_to_file:
			with open(self.get_device_funcid() + '.cu', "w") as f:
				f.write(cc)

		try:
			self.mod = cp.RawModule(code=cc)
			self.run_func = self.mod.get_function(self.get_kernel_funcid())
		except:
			with open("on_compile_fail.cu", "w") as f:
				f.write(cc)
			raise
		return cc

	def get_deps(self):
		return list()


def gain_ratio_step_funcid(nparam: int, dtype: cp.dtype):
	return 'gain_ratio_step' + ctc.dim_type_funcid(nparam, dtype)

def gain_ratio_step_code(nparam: int, dtype: cp.dtype):
	temp = Template(
"""
__device__
void {{funcid}}(const {{fp_type}}* f, const {{fp_type}}* ftp, const {{fp_type}}* pars_tp, const {{fp_type}}* step,
	const {{fp_type}}* g, const {{fp_type}}* h, {{fp_type}}* pars, 
	{{fp_type}}* lam, char* step_type, {{fp_type}} mu, {{fp_type}} eta, {{fp_type}} acc, {{fp_type}} dec, int tid, int N) 
{
	if (step_type[tid] == 0) {
		return;
	}

	{{fp_type}} actual = 0.5f * (f[tid] - ftp[tid]);
	{{fp_type}} predicted = 0.0f;

	int k = 0;
	for (int i = 0; i < {{nparam}}; ++i) {
		for (int j = 0; j <= i; ++j) {
			float entry = h[k*N+tid] * step[i*N+tid] * step[j*N+tid];
			if (i == j) {
				predicted -= entry;
			} else {
				predicted -= 2.0f * entry;
			}
			++k;
		}
	}
	predicted *= 0.5f;

	for (int i = 0; i < {{nparam}}; ++i) {
		int iidx = i*N+tid;
		predicted -= step[iidx] * g[iidx];
	}

	{{fp_type}} rho = actual / predicted;

	if ((rho > mu) && (actual > 0)) {
		for (int i = 0; i < {{nparam}}; ++i) {
			if (tid == 100) {
				printf("cp ");
			}
			int iidx = i*N+tid;
			pars[iidx] = pars_tp[iidx];
		}
		if (rho > eta) {
			lam[tid] /= acc;
			step_type[tid] = 1;
		} else {
			step_type[tid] = 2;
		}
	} else {
		lam[tid] *= dec;
		step_type[tid] = 4;
	}

	if (predicted < 0) {
		lam[tid] *= dec;
		step_type[tid] |= 8;
	}

	if (tid == 100) {
		printf("actual=%f, rho=%f, predicted=%f, step_type=%d, f=%f\\n", actual, rho, predicted, step_type[tid], f[tid]);
	}

}
""")

	type = ctc.type_to_typestr(dtype)
	funcid = gain_ratio_step_funcid(nparam, dtype)
	return temp.render(funcid=funcid, fp_type=type, nparam=nparam)

class GainRatioStep(CudaFunction):
	def __init__(self, nparam: int, dtype: cp.dtype, write_to_file: bool = False):
		self.funcid = gain_ratio_step_funcid(nparam, dtype)
		self.code = gain_ratio_step_code(nparam, dtype)

		self.write_to_file = write_to_file

		self.type_str = ctc.type_to_typestr(dtype)
		self.nparam = nparam
		self.dtype = dtype

		self.mod = None
		self.run_func = None

	def run(self, f, ftp, pars_tp, step, g, h, pars, lam, step_type, mu = 0.25, eta = 0.75, acc = 5.0, dec = 2.0):
		if self.run_func == None:
			self.build()

		N = pars.shape[1]

		Nthreads = 32
		blockSize = math.ceil(N / Nthreads)
		self.run_func((blockSize,),(Nthreads,),(f, ftp, pars_tp, step, g, h, pars, lam, step_type, 
			cp.float64(mu).astype(self.dtype), cp.float64(eta).astype(self.dtype), cp.float64(acc).astype(self.dtype), cp.float64(dec).astype(self.dtype), N))

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
void {{funcid}}(const {{fp_type}}* f, const {{fp_type}}* ftp, const {{fp_type}}* pars_tp, const {{fp_type}}* step,
	const {{fp_type}}* g, const {{fp_type}}* h, {{fp_type}}* pars, 
	{{fp_type}}* lam, char* step_type, {{fp_type}} mu, {{fp_type}} eta, {{fp_type}} acc, {{fp_type}} dec, int N) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N) {
		{{dfuncid}}(f, ftp, pars_tp, step, g, h, pars, lam, step_type, mu, eta, acc, dec, tid, N);
	}
}
""")
		fid = self.get_kernel_funcid()
		dfid = self.get_device_funcid()

		return temp.render(funcid=fid, fp_type=self.type_str, dfuncid=dfid)

	def get_full_code(self):
		code = self.get_device_code() + '\n'
		code += self.get_kernel_code()
		return code

	def build(self):
		cc = cudap.code_gen_walking(self, "")
		if self.write_to_file:
			with open(self.get_device_funcid() + '.cu', "w") as f:
				f.write(cc)
		try:
			self.mod = cp.RawModule(code=cc)
			self.run_func = self.mod.get_function(self.get_kernel_funcid())
		except:
			with open("on_compile_fail.cu", "w") as f:
				f.write(cc)
			raise
		return cc

	def get_deps(self):
		return list()


def clamp_pars_funcid(nparam: int, dtype: cp.dtype):
	return 'clamp_pars' + ctc.dim_type_funcid(nparam, dtype)

def clamp_pars_code(nparam: int, dtype: cp.dtype):
	temp = Template(
"""
__device__
void {{funcid}}(const {{fp_type}}* lower_bound, const {{fp_type}}* upper_bound, const char* step_type, {{fp_type}}* pars, int tid, int N) 
{
	if (step_type[tid] == 0) {
		return;
	}

	for (int i = 0; i < {{nparam}}; ++i) {
		int index = i*N+tid;
		{{fp_type}} p = pars[index];
		{{fp_type}} u = upper_bound[index];
		{{fp_type}} l = lower_bound[index];

		if (p > u) {
			pars[index] = u;
		} else if (p < l) {
			pars[index] = l;
		}
	}
}
""")

	type = ctc.type_to_typestr(dtype)
	funcid = clamp_pars_funcid(nparam, dtype)
	return temp.render(funcid=funcid, fp_type=type, nparam=nparam)

class ClampPars(CudaFunction):
	def __init__(self, nparam: int, dtype: cp.dtype, write_to_file: bool = False):
		
		self.funcid = clamp_pars_funcid(nparam, dtype)
		self.code = clamp_pars_code(nparam, dtype)

		self.write_to_file = write_to_file

		self.type_str = ctc.type_to_typestr(dtype)
		self.nparam = nparam
		self.dtype = dtype

		self.mod = None
		self.run_func = None

	def run(self, lower_bound, upper_bound, step_type, pars):
		if self.run_func == None:
			self.build()

		N = pars.shape[1]

		Nthreads = 32
		blockSize = math.ceil(N / Nthreads)
		self.run_func((blockSize,),(Nthreads,),(lower_bound, upper_bound, step_type, pars, N))

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
void {{funcid}}(const {{fp_type}}* lower_bound, const {{fp_type}}* upper_bound, const char* step_type, {{fp_type}}* pars, int N) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N) {
		{{dfuncid}}(lower_bound, upper_bound, step_type, pars, tid, N);
	}
}
""")
		fid = self.get_kernel_funcid()
		dfid = self.get_device_funcid()

		return temp.render(funcid=fid, fp_type=self.type_str, dfuncid=dfid)

	def get_full_code(self):
		code = self.get_device_code() + '\n'
		code += self.get_kernel_code()
		return code

	def build(self):
		cc = cudap.code_gen_walking(self, "")
		if self.write_to_file:
			with open(self.get_device_funcid() + '.cu', "w") as f:
				f.write(cc)
		try:
			self.mod = cp.RawModule(code=cc)
			self.run_func = self.mod.get_function(self.get_kernel_funcid())
		except:
			with open("on_compile_fail.cu", "w") as f:
				f.write(cc)
			raise
		return cc

	def get_deps(self):
		return list()


def gradient_convergence_funcid(nparam: int, dtype: cp.dtype):
	return 'gradient_convergence' + ctc.dim_type_funcid(nparam, dtype)

def gradient_convergence_code(nparam: int, dtype: cp.dtype):
	temp = Template(
"""
__device__
void {{funcid}}(const {{fp_type}}* pars, const {{fp_type}}* g, const {{fp_type}}* f, const {{fp_type}}* lower_bound, const {{fp_type}}* upper_bound, char* step_type, float tol, int tid, int N) 
{
	if (step_type[tid] == 0) {
		return;
	}
	
	bool clamped = false;
	{{fp_type}} clamped_norm = 0.0f;
	{{fp_type}} temp1;
	{{fp_type}} temp2;
	for (int i = 0; i < {{nparam}}; ++i) {
		int iidx = i*N+tid;
		temp1 = pars[iidx];
		temp2 = g[iidx];
		temp2 = temp1 - temp2;
		{{fp_type}} u = upper_bound[iidx];
		{{fp_type}} l = lower_bound[iidx];
		if (temp2 > u) {
			clamped = true;
			temp2 = u;
		} else if (temp2 < l) {
			clamped = true;
			temp2 = l;
		}
		temp2 = temp1 - temp2;
		clamped_norm += temp2*temp2;
	}

	if (clamped_norm < tol*(1 + f[tid])) {
		if ((step_type[tid] & 1) || clamped) {
			step_type[tid] = 0;
		}
	}
}
""")

	type = ctc.type_to_typestr(dtype)
	funcid = gradient_convergence_funcid(nparam, dtype)
	return temp.render(funcid=funcid, fp_type=type, nparam=nparam)
	
class GradientConvergence(CudaFunction):
	def __init__(self, nparam: int, dtype: cp.dtype, write_to_file: bool = False):
		self.funcid = gradient_convergence_funcid(nparam, dtype)
		self.code = gradient_convergence_code(nparam, dtype)

		self.write_to_file = write_to_file

		self.type_str = ctc.type_to_typestr(dtype)
		self.nparam = nparam
		self.dtype = dtype

		self.mod = None
		self.run_func = None

	def run(self, pars, g, f, lower_bound, upper_bound, step_type, tol):
		if self.run_func == None:
			self.build()

		N = pars.shape[1]

		Nthreads = 32
		blockSize = math.ceil(N / Nthreads)
		self.run_func((blockSize,),(Nthreads,),(pars, g, f, lower_bound, upper_bound, step_type, cp.float64(tol).astype(self.dtype), N))

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
void {{funcid}}(const float* pars, const float* g, const {{fp_type}}* f, const {{fp_type}}* lower_bound, const {{fp_type}}* upper_bound, char* step_type, float tol, int N) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N) {
		{{dfuncid}}(pars, g, f, lower_bound, upper_bound, step_type, tol, tid, N);
	}
}
""")
		fid = self.get_kernel_funcid()
		dfid = self.get_device_funcid()

		return temp.render(funcid=fid, fp_type=self.type_str, dfuncid=dfid)

	def get_full_code(self):
		code = self.get_device_code() + '\n'
		code += self.get_kernel_code()
		return code

	def build(self):
		cc = cudap.code_gen_walking(self, "")
		if self.write_to_file:
			with open(self.get_device_funcid() + '.cu', "w") as f:
				f.write(cc)

		try:
			self.mod = cp.RawModule(code=cc)
			self.run_func = self.mod.get_function(self.get_kernel_funcid())
		except:
			with open("on_compile_fail.cu", "w") as f:
				f.write(cc)
			raise
		return cc

	def get_deps(self):
		return list()


class SecondOrderLevenbergMarquardt(CudaFunction):
	def __init__(self, expr: str, pars_str: list[str], consts_str: list[str], ndata: int, dtype: cp.dtype, write_to_file: bool = False):
		self.nparam = len(pars_str)
		self.nhes = round(self.nparam * (self.nparam + 1) / 2)
		self.nconst = len(consts_str)
		self.ndata = ndata
		self.dtype = dtype

		self.write_to_file = write_to_file

		self.expr = expr
		self.pars_str = pars_str.copy()
		self.consts_str = consts_str.copy()

		self.gradcu = ResJacGradHesHesl(self.expr, self.pars_str.copy(), self.consts_str.copy(), self.ndata, self.dtype, self.write_to_file)
		self.gradcu.build()
		self.gradsumcu = SumResGradHesHesl(self.nparam, self.ndata, self.dtype, self.write_to_file)
		self.gradsumcu.build()
		self.gmw81solcu = solver.GMW81Solver(self.nparam, self.dtype, self.write_to_file)
		self.gmw81solcu.build()
		self.rescu = ResF(self.expr, self.pars_str.copy(), self.consts_str.copy(), self.ndata, self.dtype, self.write_to_file)
		self.rescu.build()
		self.gaincu = GainRatioStep(self.nparam, self.dtype, self.write_to_file)
		self.gaincu.build()
		self.clampcu = ClampPars(self.nparam, self.dtype, self.write_to_file)
		self.clampcu.build()
		self.convcu = GradientConvergence(self.nparam, self.dtype, self.write_to_file)
		self.convcu.build()

		self.batch_size = int(1)
		self.Nelem = self.batch_size * self.ndata

	def setup(self, pars_t, consts_t, data_t, lower_bound_t, upper_bound_t):
		self.batch_size = pars_t.shape[1]
		self.Nelem = data_t.shape[1]
		if round(self.Nelem / self.batch_size) != self.ndata:
			raise RuntimeError('Nelem to batch_size ratio does not equal ndata')

		self.pars_t = pars_t
		self.consts_t = consts_t
		self.data_t = data_t
		self.lower_bound_t = lower_bound_t
		self.upper_bound_t = upper_bound_t

		self.first_f = (cp.finfo(cp.float32).max / 10.0)*cp.ones((1, self.batch_size), dtype=self.dtype)
		self.last_f = (cp.finfo(cp.float32).max / 10.0)*cp.ones((1, self.batch_size), dtype=self.dtype)

		self.lam_t = 1*cp.ones((1, self.batch_size), dtype=self.dtype)
		self.step_t = cp.empty((self.nparam, self.batch_size), dtype=self.dtype)
		self.res_t = cp.empty((1, self.Nelem), dtype=self.dtype)
		self.jac_t = cp.empty((self.nparam, self.Nelem), dtype=self.dtype)
		self.grad_t = cp.empty((self.nparam, self.Nelem), dtype=self.dtype)
		self.hes_t = cp.empty((self.nhes, self.Nelem), dtype=self.dtype)
		self.hesl_t = cp.empty((self.nhes, self.Nelem), dtype=self.dtype)
		self.step_type_t = cp.ones((1, self.batch_size), dtype=cp.int8)
		self.f_t = cp.empty((1, self.batch_size), dtype=self.dtype)
		self.ftp_t = cp.empty((1, self.batch_size), dtype=self.dtype)
		self.g_t = cp.empty((self.nparam, self.batch_size), dtype=self.dtype)
		self.h_t = cp.empty((self.nhes, self.batch_size), dtype=self.dtype)
		self.hl_t = cp.empty((self.nhes, self.batch_size), dtype=self.dtype)
		self.pars_tp_t = cp.empty((self.nparam, self.batch_size), dtype=self.dtype)

	def run(self, iters: int, tol: float):

		for i in range(0,iters):
			self.f_t.fill(0.0)
			self.ftp_t.fill(0.0)
			self.g_t.fill(0.0)
			self.h_t.fill(0.0)
			self.hl_t.fill(0.0)

			self.gradcu.run(self.pars_t, self.consts_t, self.data_t, self.lam_t, self.step_type_t,
				self.res_t, self.jac_t, self.grad_t, self.hes_t, self.hesl_t)
			self.gradsumcu.run(self.res_t, self.grad_t, self.hes_t, self.hesl_t, self.step_type_t,
				self.f_t, self.g_t, self.h_t, self.hl_t)
			self.gmw81solcu.run(self.hl_t, self.g_t, self.step_type_t, self.step_t)

			if i == 0:
				self.first_f = self.f_t.copy()
			if i == iters-1:
				self.last_f = self.f_t.copy()

			self.step_t = cp.nan_to_num(self.step_t, copy=False, posinf=0.0, neginf=0.0)
			#self.pars_tp_t = self.pars_t - self.step_t
			cp.subtract(self.pars_t, self.step_t, out=self.pars_tp_t)
			#cp.add(self.pars_t, self.step_t, out=self.pars_tp_t)
			#self.pars_t -= np.float32(((i+1) / iters))*self.step_t

			self.rescu.run(self.pars_tp_t, self.consts_t, self.data_t, self.step_type_t, self.res_t, self.ftp_t)
			self.gaincu.run(self.f_t, self.ftp_t, self.pars_tp_t, self.step_t, 
			   self.g_t, self.h_t, self.pars_t, self.lam_t, self.step_type_t)

			#self.clampcu.run(self.lower_bound_t, self.upper_bound_t, self.step_type_t, self.pars_t)
			
			#self.convcu.run(self.pars_t, self.g_t, self.f_t, self.lower_bound_t, 
			#   self.upper_bound_t, self.step_type_t, cp.float32(tol))



