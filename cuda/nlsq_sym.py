
from ..sym import sym

import cupy as cp
from jinja2 import Template

from .cuda_program import CudaTensorChecking as ctc
from .cuda_program import CudaTensor, CudaFunction

from ..cuda import linalg
from ..cuda import solver

def nlsq_res_jac_hes_lhes_funcid(expr: str, ndata: int, nparam: int, nconst: int, dtype: cp.dtype):
    return 'nlsq_res_jac_hes_lhes' + ctc.dim_dim_dim_type_funcid(ndata, nparam, nconst, 
        dtype, 'nlsq_res_jac_hes_lhes') + sym.expr_hash(expr, 12)

def nlsq_res_jac_hes_lhes_code(expr: str, pars: list[str], consts: list[str], 
        ndata: int, nparam: int, nconst: int, dtype: cp.dtype):
    
    rjh_temp = Template(
"""
__device__
void {{funcid}}(const {{fp_type}}* params, const {{fp_type}}* consts, 
	const {{fp_type}}* data, const {{fp_type}}* weights, float lambda, {{fp_type}}* res, 
	{{fp_type}}* jac, {{fp_type}}* hes, {{fp_type}}* lam_hes) 
{
	{{zero_mat_funcid}}(hes);

	for (int i = 0; i < {{ndata}}; ++i) {
{{sub_expr}}

{{res_expr}}

{{jac_expr}}

{{hes_expr}}
	}

	for (int i = 1; i < {{npar}}; ++i) {
		for (int j = 0; j < i; ++j) {
			hes[j*{{npar}} + i] = hes[i*{{npar}} + j];
		}
	}

	{{mul_transpose_diag_mat_funcid}}(jac, weights, lam_hes);
	{{add_mat_mat_ldiag_funcid}}(hes, lamda, lam_hes);
}
""")

    type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'nlsq_res_jac_hes_lhes')

    funcid = nlsq_res_jac_hes_lhes_funcid(expr, ndata, nparam, nconst, dtype)

    substs, reduced = sym.res_jac_hes(expr, pars, consts)
    cuprint = sym.CUDAPrinter()

    sub_str = ""
    for s in substs:
        sub_str += '\t\t'+type+' '+cuprint.tcs_f(s[0])+' = '+cuprint.tcs_f(s[1])+';\n'

    res_str = ""
    for s in reduced[:1]:
        res_str += '\t\tres[i] = '+cuprint.tcs_f(s)+';'

    jac_str = ""
    i = 0
    for s in reduced[1:(len(pars) + 1)]:
        jac_str += '\t\tjac[i*'+str(len(pars))+'+'+str(i)+'] = '+cuprint.tcs_f(s)+';\n'
        i += 1

    hes_str = ""
    k = 0
    for i in range(0,len(pars)):
        for j in range(0,i+1):
            s = reduced[len(pars) + k + 1]
            hes_str += '\t\thes['+str(i)+'*'+str(len(pars))+'+'+str(j)+'] += '+cuprint.tcs_f(s)+';\n'
            k += 1

    zm_fid = linalg.zero_mat_funcid(nparam, nparam, dtype)
    mtdm_fid = linalg.mul_transpose_diag_mat_funcid(ndata, nparam, dtype)
    amml_fid = linalg.add_mat_mat_ldiag_funcid(nparam, dtype)

    rjh_kernel = rjh_temp.render(funcid=funcid, fp_type=type, ndata=str(ndata), 
        nparam=str(nparam), nconst=str(nconst), zero_mat_funcid=zm_fid,
        sub_expr=sub_str, res_expr=res_str, jac_expr=jac_str, hes_expr=hes_str,
        mul_transpose_diag_mat_funcid=mtdm_fid, add_mat_mat_ldiag_funcid=amml_fid)

    return rjh_kernel

class NLSQResJacHesLHes(CudaFunction):
    def __init__(self):
        super().__init__()