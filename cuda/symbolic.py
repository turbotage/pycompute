
import cupy as cp
from jinja2 import Template
from symengine import sympify

#from ..sym import sym
from sym import util

from . import linalg
from .cuda_program import CudaFunction, CudaTensor
from .cuda_program import CudaTensorChecking as ctc


def eval_jac_hes_funcid(expr: str, pars_str: list[str], consts_str: list[str], nelem: int, dtype: cp.dtype):
	return 'eval_jac_hes' + ctc.dim_dim_dim_type_funcid(nelem, len(pars_str), len(consts_str), 
		dtype, 'eval_jac_hes') + '_' + util.expr_hash(expr, 12)

def eval_jac_hes_code(expr: str, pars_str: list[str], consts_str: list[str], nelem: int, dtype: cp.dtype):
	
	rjh_temp = Template(
"""
void {{funcid}}(const {{fp_type}}* params, const {{fp_type}}* consts, {{fp_type}}* eval, 
	{{fp_type}}* jac, {{fp_type}}* hes, unsigned int N) 
{
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N) {

		{{fp_type}} pars[{{nparam}}];
		for (int i = 0; i < {{nparam}}; ++i) {
			pars[i] = params[i*{{nelem}}+tid];
		}

{{sub_expr}}

{{eval_expr}}

{{jac_expr}}

{{hes_expr}}

	}


}
""")

	type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'eval_jac_hes')

	nparam = len(pars_str)
	nconst = len(consts_str)

	funcid = eval_jac_hes_funcid(expr, pars_str, consts_str, nelem, dtype)

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
		sub_str += '\t\t'+type+' '+cuprint.tcs_f(s[0])+' = '+cuprint.tcs_f(s[1])+';\n'

	eval_str = '\t\teval[tid] = '+cuprint.tcs_f(reduced[0])+';'

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
		jac_str += '\t\tjac['+str(k)+'*N+tid] = '+ctstr+';\n'

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
		hes_str += '\t\thes['+str(k)+'*N+tid] = '+ctstr+';\n'

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
		nparam=nparam, nconst=nconst, nelem=nelem,
		sub_expr=sub_str, eval_expr=eval_str, jac_expr=jac_str, hes_expr=hes_str)


	return rjh_kernel

class EvalJacHes(CudaFunction):
	def __init__(self, expr: str, pars_str: list[str], consts_str: list[str],
		pars: CudaTensor, consts: CudaTensor,
		eval: CudaTensor, jac: CudaTensor, hes: CudaTensor):

		type_str = ctc.check_fp32_or_fp64(pars, 'eval_jac_hes')

		self.pars = pars
		self.consts = consts
		self.eval = eval
		self.jac = jac
		self.hes = hes

		self.nelem = pars.shape[1]
		self.type_str = type_str
		self.pars_str = pars_str
		self.consts_str = consts_str

		self.funcid = eval_jac_hes_funcid(expr, pars_str, consts_str, self.nelem, self.pars.dtype)
		self.code = eval_jac_hes_code(expr, pars_str, consts_str, self.nelem, self.pars.dtype)

	def get_funcid(self):
		return self.funcid

	def get_device_code(self):
		return "__device__" + self.code

	def get_kernel_code(self):
		return "extern \"C\" __global__" + self.code

	def get_deps(self):
		return list()


