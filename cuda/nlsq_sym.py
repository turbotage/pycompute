
import cupy as cp
from jinja2 import Template
from symengine import sympify

#from ..sym import sym
from sym import util

from . import linalg
from .cuda_program import CudaFunction, CudaTensor
from .cuda_program import CudaTensorChecking as ctc


def nlsq_res_jac_hes_lhes_funcid(expr: str, ndata: int, nparam: int, nconst: int, dtype: cp.dtype):
	return 'nlsq_res_jac_hes_lhes' + ctc.dim_dim_dim_type_funcid(ndata, nparam, nconst, 
		dtype, 'nlsq_res_jac_hes_lhes') + '_' + util.expr_hash(expr, 12)

def nlsq_res_jac_hes_lhes_code(expr: str, pars_str: list[str], consts_str: list[str], 
		ndata: int, nparam: int, nconst: int, dtype: cp.dtype):
	
	rjh_temp = Template(
"""
__device__
void {{funcid}}(const {{fp_type}}* params, const {{fp_type}}* consts, 
	const {{fp_type}}* data, {{fp_type}} lambda, {{fp_type}}* res, 
	{{fp_type}}* jac, {{fp_type}}* hes, {{fp_type}}* lhes) 
{
	{{zero_mat_funcid}}(hes);

	for (int i = 0; i < {{ndata}}; ++i) {
{{sub_expr}}

{{res_expr}}

{{jac_expr}}

{{hes_expr}}
	}

	for (int i = 1; i < {{nparam}}; ++i) {
		for (int j = 0; j < i; ++j) {
			hes[j*{{nparam}}+i] = hes[i*{{nparam}} + j];
		}
	}

	{{mul_transpose_mat_funcid}}(jac, lhes);
	{{add_mat_mat_ldiag_funcid}}(hes, lambda, lhes);
}
""")

	type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'nlsq_res_jac_hes_lhes')

	funcid = nlsq_res_jac_hes_lhes_funcid(expr, ndata, nparam, nconst, dtype)

	sym_expr = sympify(expr)
	# convert parameter names to ease kernel generation
	for i in range(0,len(pars_str)):
		temp = pars_str[i]
		pars_str[i] = 'parvar_' + temp
		sym_expr = sym_expr.subs(temp, pars_str[i])

	for i in range(0,len(consts_str)):
		temp = consts_str[i]
		consts_str[i] = 'convar_' + temp
		sym_expr = sym_expr.subs(temp, consts_str[i])

	substs, reduced = util.res_jac_hes(str(sym_expr), pars_str, consts_str)
	cuprint = util.CUDAPrinter()

	sub_str = ""
	for s in substs:
		sub_str += '\t\t'+type+' '+cuprint.tcs_f(s[0])+' = '+cuprint.tcs_f(s[1])+';\n'

	res_str = ""
	for s in reduced[:1]:
		res_str += '\t\tres[i] = '+cuprint.tcs_f(s)+' - data[i];'

	jac_str = ""
	i = 0
	for s in reduced[1:(len(pars_str) + 1)]:
		jac_str += '\t\tjac[i*'+str(len(pars_str))+'+'+str(i)+'] = '+cuprint.tcs_f(s)+';\n'
		i += 1

	hes_str = ""
	k = 0
	for i in range(0,len(pars_str)):
		for j in range(0,i+1):
			s = reduced[len(pars_str) + k + 1]
			hes_str += '\t\thes['+str(i)+'*'+str(len(pars_str))+'+'+str(j)+'] += '+cuprint.tcs_f(s)+' * res[i];\n'
			k += 1

	i = 0
	for p in pars_str:
		repl = 'params['+str(i)+']'
		sub_str = sub_str.replace(p, repl)
		res_str = res_str.replace(p, repl)
		jac_str = jac_str.replace(p, repl)
		hes_str = hes_str.replace(p, repl)

	i = 0
	for c in consts_str:
		repl = 'consts[i*'+str(nconst)+'+'+str(i)+']'
		sub_str = sub_str.replace(c, repl)
		res_str = res_str.replace(c, repl)
		jac_str = jac_str.replace(c, repl)
		hes_str = hes_str.replace(c, repl)

	zm_fid = linalg.zero_mat_funcid(nparam, nparam, dtype)
	mtm_fid = linalg.mul_transpose_mat_funcid(ndata, nparam, dtype)
	amml_fid = linalg.add_mat_mat_ldiag_funcid(nparam, dtype)

	rjh_kernel = rjh_temp.render(funcid=funcid, fp_type=type, ndata=ndata,
		nparam=nparam, nconst=nconst, zero_mat_funcid=zm_fid,
		sub_expr=sub_str, res_expr=res_str, jac_expr=jac_str, hes_expr=hes_str,
		mul_transpose_mat_funcid=mtm_fid, add_mat_mat_ldiag_funcid=amml_fid)


	return rjh_kernel

class NLSQResJacHesLHes(CudaFunction):
	def __init__(self, expr: str, pars_str: list[str], consts_str: list[str],
		pars: CudaTensor, consts: CudaTensor, data: CudaTensor,
		res: CudaTensor, jac: CudaTensor, hes: CudaTensor, lhes: CudaTensor):

		ctc.check_fp32_or_fp64(res, 'nlsq_res_jac_hes_lhes')
		ctc.check_fp32_or_fp64(jac, 'nlsq_res_jac_hes_lhes')
		ctc.check_fp32_or_fp64(hes, 'nlsq_res_jac_hes_lhes')
		type = ctc.check_fp32_or_fp64(lhes, 'nlsq_res_jac_hes_lhes')
		ctc.check_is_vec(pars, 'nlsq_res_jac_hes_lhes')
		ctc.check_is_vec(res, 'nlsq_res_jac_hes_lhes')
		ctc.check_square_mat(hes, 'nlsq_res_jac_hes_lhes')
		ctc.check_square_mat(lhes, 'nlsq_res_jac_hes_lhes')
		ctc.check_is_same_shape(hes, lhes, 'nlsq_res_jac_hes_lhes')

		if jac.shape[1] != res.shape[1]:
			raise RuntimeError('dim1 of jac must be same as dim1 or res')

		if jac.shape[2] != hes.shape[1]:
			raise RuntimeError('dim1 of jac must be same as hes dim')

		if consts.shape[1] != res.shape[1]:
			raise RuntimeError('dim1 of consts must be same as dim1 or res')

		self.ndata = res.shape[1]
		self.nparam = jac.shape[2]
		self.nconst = consts.shape[2]
		self.type_str = type

		self.pars = pars
		self.consts = consts
		self.data = data
		self.res = res
		self.jac = jac
		self.hes = hes
		self.lhes = lhes

		self.funcid = nlsq_res_jac_hes_lhes_funcid(expr, self.ndata, self.nparam, self.nconst, self.hes.dtype)
		self.code = nlsq_res_jac_hes_lhes_code(expr, pars_str, 
			consts_str, self.ndata, self.nparam, self.nconst, self.hes.dtype)

	def get_funcid(self):
		return self.funcid

	def get_code(self):
		return self.code

	def get_deps(self):
		return [linalg.ZeroMat(self.hes), linalg.MulTransposeMat(self.jac), linalg.AddMatMatLdiag(self.hes)]
		
	def get_batched_kernel(self):
		kernel_code = Template(
"""
extern "C" __global__
void {{funcid}}(const {{fp_type}}* params, const {{fp_type}}* consts, const {{fp_type}}* data, 
	{{fp_type}} lambda, {{fp_type}}* res, {{fp_type}}* jac, {{fp_type}}* hes, {{fp_type}}* lhes, unsigned int N) {
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N) {
		unsigned int par_id = {{nparam}} * tid;
		unsigned int con_id = {{ndata}} * {{nconst}} * tid;
		unsigned int res_id = {{ndata}} * tid;
		unsigned int jac_id = {{ndata}} * {{nparam}} * tid;
		unsigned int hes_id = {{nparam}} * {{nparam}} * tid;
		
		{{nlsq_rjhlh_funcid}}(&params[par_id], &consts[con_id], &data[res_id], lambda,
			&res[res_id], &jac[jac_id], &hes[hes_id], &lhes[hes_id]);
	}
}
""")
		nlsq_fid = self.get_funcid()
		fid = 'bk_' + nlsq_fid

		return kernel_code.render(funcid=fid, fp_type=self.type_str, ndata=self.ndata, nconst=self.nconst,
			nparam=self.nparam, nlsq_rjhlh_funcid=nlsq_fid)


