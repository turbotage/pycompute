
from jinja2 import Template
import cupy as cp
from enum import Enum
#from numba import jit

from cuda_program import CudaFunction, CudaTensor

class Linkage(Enum):
	GLOBAL = 1
	LOCAL = 2


def dim_type_funcid(ndim: int, dtype: cp.dtype):
	type_qualifier: str
	if dtype == cp.float32:
		type_qualifier = 'f'
	elif dtype == cp.float64:
		type_qualifier = 'd'
	else:
		raise RuntimeError('Does only support fp32 and fp64')

	return '_' + ndim + '_' + type_qualifier


def ldl_funcid(ndim: int, dtype: cp.dtype):
	return 'ldl' + dim_type_funcid(ndim, dtype)

def ldl_code(ndim: int, dtype: cp.dtype, linkage: Linkage):
	codestr = Template("""
	void {{funcid}}({{fp_type}}* mat) {
		{{fp_type}} arr[{{ndim}}];
		for (int i = 0; i < ndim; ++i) {
			{{fp_type}} d = mat[i*{{ndim}} + i];

			for (int j = i + 1; j < {{ndim}}; ++j) {
				arr[j] = mat[j*{{ndim}} + i];
				mat[j*{{ndim}} + i] /= d;
			}

			for (int j = i + 1; j < {{ndim}}; ++j) {
				{{fp_type}} aj = arr[j];
				for (int k = j; k < ndim; ++k) {
					mat[k*{{ndim}} + j] -= aj * mat[k*{{ndim}} + i];
				}
			}
		}
	}
	""")

	type: str
	if dtype == cp.float32:
		type = 'float'
	elif dtype == cp.float64:
		type = 'double'
	else:
		raise RuntimeError('LDL does only support fp32 and fp64')

	funcid = ldl_funcid(ndim, dtype)

	return codestr.render(funcid=funcid, fp_type=type, ndim=ndim)

class LDL(CudaFunction):
	def __init__(self, mat: CudaTensor):
		if len(mat.shape) != 3:
			raise RuntimeError('Shapes array must be 3 long')
		if mat.shape[1] == mat.shape[2]:
			raise RuntimeError('Matrix must be square')

		




def gmw81_funcid(ndim: int, dtype: cp.dtype):
	return 'gmw81' + dim_type_funcid(ndim, dtype)

def gmw81_code(ndim: int, dtype: cp.dtype, linkage: Linkage):
	codestr = Template("""
	void {{funcid}}({{fp_type}}* mat) {
		{{fp_type}} m1 = 0.0f;
		{{fp_type}} m2 = 0.0f;
		{{fp_type}} beta2 = 0.0f;
		{{fp_type}} temp;
		{{fp_type}} arr[{{ndim}}];

		for (int i = 0; i < {{ndim}}; ++i) {
			temp = {{abs_func}}(mat[i*{{ndim}}+i]);
			if (m1 < temp) {
				m1 = temp;
			}
		}

		if (beta2 < m1) {
			beta2 = m1;
		}

		for (int i = 1; i < {{ndim}}; ++i) {
			for (int j = 0; j < i; ++j) {
				temp = {{abs_func}}(mat[i*ndim + j]);
				if (m2 < temp) {
					m2 = temp;
				}
			}
		}

		if ({{ndim}} > 1) {
			m2 /= {{sqrt_func}}({{ndim}}*{{ndim}} - 1);
		}

		if (beta2 < m2) {
			beta2 = m2;
		}

		for (int i = 0; i < {{ndim}}; ++i) {
			{{fp_type}} d = {{abs_type}}(mat[i*{{ndim}} + i]);

			if (d < {{machine_eps}}) {
				d = {{machine_eps}};
			}

			m2 = 0.0f;
			for (int j = i + 1; j < {{ndim}}; ++j) {
				temp = {{abs_func}}(mat[j*{{ndim}} + i]);
				if (m2 < temp) {
					m2 = temp;
				}
			}

			m2 *= m2;

			if (m2 > d * beta2) {
				d = m2 / beta2;
			}

			mat[i*{{ndim}} + i] = d;

			for (int j = i + 1; j < {{ndim}}; ++j) {
				arr[j] = mat[j*{{ndim}} + i];
				mat[j*{{ndim}} + i] /= d;
			}

			for (int j = i + 1; j < {{ndim}}; ++j) {
				for (int k = j; k < {{ndim}}; ++k) {
					mat[k*{{ndim}} + j] -= arr[j] * mat[k*{{ndim}} + i];
				}
			}

		}

	}
	""")

	type: str
	abs_func: str
	sqrt_func: str
	machine_eps: str
	if dtype == cp.float32:
		type = 'float'
		abs_func = 'fabsf'
		sqrt_func = 'sqrtf'
		machine_eps = '1e-6'
	elif dtype == cp.float64:
		type = 'double'
		abs_func = 'fabs'
		sqrt_func = 'sqrt'
		machine_eps = '1e-15'
	else:
		raise RuntimeError('gmw81 does only support fp32 and fp64')

	funcid = gmw81_funcid(ndim, dtype)

	return codestr.render(funcid=funcid, fp_type=type, ndim=ndim, 
		abs_func=abs_func, sqrt_func=sqrt_func, machine_eps=machine_eps)




def forward_subs_unit_diaged_funcid(ndim: int, dtype: cp.dtype):
	return 'forward_subs_unit_diaged' + dim_type_funcid(ndim, dtype)

def forward_subs_unit_diaged_code(ndim: int, dtype: cp.dtype):
	codestr = Template("""
	void {{funcid}}({{fp_type}}* mat, {{fp_type}}* rhs, {{fp_type}}* sol) {
		for (int i = 0; i < ndim; ++i) {
			sol[i] = rhs[i];
			for (int j = 0; j < i; ++j) {
				sol[i] = -= mat[i*{{ndim}} + j] * mat[j*{{ndim}} + j] * sol[j];
			}
			sol[i] /= mat[i*{{ndim}} + i];
		}
	}
	""")

	type: str
	if dtype == cp.float32:
		type = 'float'
	elif dtype == cp.float64:
		type = 'double'
	else:
		raise RuntimeError('LDL does only support fp32 and fp64')

	funcid = forward_subs_unit_diaged_funcid(ndim, dtype)

	return codestr.render(funcid=funcid, fp_type=type, ndim=ndim)



def backward_subs_unit_t_funcid(ndim: int, dtype: cp.dtype):
	return 'backward_subs_unit_t' + dim_type_funcid(ndim, dtype)

def backward_subs_unit_t_code(ndim: int, dtype: cp.dtype):
	codestr = Template("""
	void {{funcid}}({{fp_type}}* mat, {{fp_type}}* rhs, {{fp_type}}* sol) {
		for (int i = {{ndim}} - 1; i >= 0; --i) {
			sol[i] = rhs[i];
			for (int j = i + 1; j < {{ndim}}; ++j) {
				sol[i] -= mat[j*{{ndim}} + i] * sol[j];
			}
		}
	}
	""")

	type: str
	if dtype == cp.float32:
		type = 'float'
	elif dtype == cp.float64:
		type = 'double'
	else:
		raise RuntimeError('LDL does only support fp32 and fp64')

	funcid = backward_subs_unit_t_funcid(ndim, dtype)

	return codestr.render(funcid=funcid, fp_type=type, ndim=ndim)
