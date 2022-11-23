
from jinja2 import Template
import cupy as cp
#from numba import jit

import math

from . import cuda_program as cudap
from .cuda_program import CudaFunction, CudaTensor
from .cuda_program import CudaTensorChecking as ctc

from . import permute
from . import linalg


# P @ (A + E) @ P^T = L @ D @ L^T
def gmw81_funcid(ndim: int, dtype: cp.dtype):
	return 'gmw81' + ctc.dim_type_funcid(ndim, dtype)

# def gmw81_code(ndim: int, dtype: cp.dtype):
# 	codestr = Template(
# """
# __device__
# void {{funcid}}({{fp_type}}* mat) {
# 	{{fp_type}} m1 = 0.0f;
# 	{{fp_type}} m2 = 0.0f;
# 	{{fp_type}} beta2 = 0.0f;
# 	{{fp_type}} temp;
# 	{{fp_type}} arr[{{ndim}}];

# 	for (int i = 0; i < {{ndim}}; ++i) {
# 		temp = {{abs_func}}(mat[i*{{ndim}}+i]);
# 		if (m1 < temp) {
# 			m1 = temp;
# 		}
# 	}

# 	if (beta2 < m1) {
# 		beta2 = m1;
# 	}

# 	for (int i = 1; i < {{ndim}}; ++i) {
# 		for (int j = 0; j < i; ++j) {
# 			temp = {{abs_func}}(mat[i*{{ndim}}+j]);
# 			if (m2 < temp) {
# 				m2 = temp;
# 			}
# 		}
# 	}

# 	if ({{ndim}} > 1) {
# 		m2 /= {{sqrt_func}}({{ndim}}*{{ndim}} - 1);
# 	}

# 	if (beta2 < m2) {
# 		beta2 = m2;
# 	}

# 	for (int i = 0; i < {{ndim}}; ++i) {
# 		{{fp_type}} d = {{abs_type}}(mat[i*{{ndim}}+i]);

# 		if (d < {{machine_eps}}) {
# 			d = {{machine_eps}};
# 		}

# 		m2 = 0.0f;
# 		for (int j = i + 1; j < {{ndim}}; ++j) {
# 			temp = {{abs_func}}(mat[j*{{ndim}}+i]);
# 			if (m2 < temp) {
# 				m2 = temp;
# 			}
# 		}

# 		m2 *= m2;

# 		if (m2 > d * beta2) {
# 			d = m2 / beta2;
# 		}

# 		mat[i*{{ndim}}+i] = d;

# 		for (int j = i + 1; j < {{ndim}}; ++j) {
# 			int ji = j*{{ndim}}+i;
# 			arr[j] = mat[ji];
# 			mat[ji] /= d;
# 		}

# 		for (int j = i + 1; j < {{ndim}}; ++j) {
# 			for (int k = j; k < {{ndim}}; ++k) {
# 				mat[k*{{ndim}}+j] -= arr[j] * mat[k*{{ndim}}+i];
# 			}
# 		}

# 	}

# }
# """)

# 	type: str = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'gmw81')
# 	abs_func: str
# 	sqrt_func: str
# 	machine_eps: str
# 	if dtype == cp.float32:
# 		abs_func = 'fabsf'
# 		sqrt_func = 'sqrtf'
# 		machine_eps = '5e-7'
# 	else:
# 		abs_func = 'fabs'
# 		sqrt_func = 'sqrt'
# 		machine_eps = '5e-16'


# 	funcid = gmw81_funcid(ndim, dtype)

# 	return codestr.render(funcid=funcid, fp_type=type, ndim=ndim,
# 		abs_func=abs_func, sqrt_func=sqrt_func, machine_eps=machine_eps)

def gmw81_code(ndim: int, dtype: cp.dtype):
	codestr = Template(
"""
__device__
void {{funcid}}({{fp_type}}* mat) {
	{{fp_type}} t0;
	{{fp_type}} t1 = 0.0f; // gamma
	{{fp_type}} t2 = 0.0f; // nu
	{{fp_type}} beta2 = {{machine_eps}};
	{{fp_type}} delta = {{machine_eps}};
	{{fp_type}} arr[{{ndim}}];

	for (int i = 0; i < {{ndim}}; ++i) {
		for (int j = 0; j <= i; ++j) {
			t0 = {{abs_func}}(mat[i*{{ndim}}+i]);
			if (i == j) {
				if (t0 > t1) {
					t1 = t0;
				}
			} else {
				if (t0 > t2) {
					t2 = t0;
				}
			}
		}
	}

	if ({{ndim}} > 1) {
		t2 /= {{sqrt_func}}({{ndim}}*{{ndim}} - 1);
	}


	if (beta2 < t1) {
		beta2 = t1;
	}
	if (beta2 < t2) {
		beta2 = t2;
	}
	t0 = t1 + t2;
	if (t0 > 1.0) {
		delta *= t0;
	}
	// delta = eps*max(gamma + nu, 1)
	// beta2 = max(gamma, nu/sqrt(n^^2-1), eps)

	for (int i = 0; i < {{ndim}}; ++i) {
		{{fp_type}} d = {{abs_type}}(mat[i*{{ndim}}+i]);

		t1 = 0.0f;
		for (int j = i + 1; j < {{ndim}}; ++j) {
			t0 = {{abs_func}}(mat[j*{{ndim}}+i]);
			if (t1 < t0) {
				t1 = t0;
			}
		}
		t1 *= t1; // t1 holds theta

		t0 = t1 / beta2;
		if (d < t0) {
			d = t0;
		}
		if (d < delta) {
			d = delta;
		}

		mat[i*{{ndim}}+i] = d;

		for (int j = i + 1; j < {{ndim}}; ++j) {
			int ji = j*{{ndim}}+i;
			arr[j] = mat[ji];
			mat[ji] /= d;
		}

		for (int j = i + 1; j < {{ndim}}; ++j) {
			for (int k = j; k < {{ndim}}; ++k) {
				mat[k*{{ndim}}+j] -= arr[j] * mat[k*{{ndim}}+i];
			}
		}

	}

}
""")

	type: str = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'gmw81')
	abs_func: str
	sqrt_func: str
	machine_eps: str
	if dtype == cp.float32:
		abs_func = 'fabsf'
		sqrt_func = 'sqrtf'
		machine_eps = '2e-7'
	else:
		abs_func = 'fabs'
		sqrt_func = 'sqrt'
		machine_eps = '4e-16'


	funcid = gmw81_funcid(ndim, dtype)

	return codestr.render(funcid=funcid, fp_type=type, ndim=ndim,
		abs_func=abs_func, sqrt_func=sqrt_func, machine_eps=machine_eps)


class GMW81(CudaFunction):
	def __init__(self, ndim: int, dtype: cp.dtype):
		self.funcid = gmw81_funcid(ndim, dtype)
		self.code = gmw81_code(ndim, dtype)
		self.type_str = ctc.type_to_typestr(dtype)

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
void {{funcid}}({{fp_type}}* mat, int N) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N) {
		{{fp_type}} mat_copy[{{ndim}}*{{ndim}}];
		int k = 0;
		for (int i = 0; i < {{ndim}}; ++i) {
			for (int j = 0; i <= j; ++j) {
				{{fp_type}} temp = mat[k*N+tid];
				mat_copy[i*{{ndim}}+j] = temp;
				mat_copy[j*{{ndim}}+i] = temp;
				++k;
			}
		}

		{{dfuncid}}(mat_copy);

		k = 0;
		for (int i = 0; i < {{ndim}}; ++i) {
			for (int j = 0; j <= i; ++j) {
				mat[k*{{ndim}}+tid] = mat_copy[i*{{ndim}}+j];
				++k;
			}
		}
	}
}
""")
		fid = self.get_kernel_funcid()
		dfid = self.get_device_funcid()

		ndim=self.ndim

		return temp.render(funcid=fid, dfuncid=dfid, fp_type=self.type_str,ndim=ndim)

	def get_deps(self):
		return list()

# FORWARD AND BACKWARD SUBSTITUTIONS #

def forward_subs_unit_diaged_funcid(ndim: int, dtype: cp.dtype):
	return 'forward_subs_unit_diaged' + ctc.dim_type_funcid(ndim, dtype)

def forward_subs_unit_diaged_code(ndim: int, dtype: cp.dtype):
	codestr = Template(
"""
__device__
void {{funcid}}(const {{fp_type}}* mat, const {{fp_type}}* rhs, {{fp_type}}* sol) {
	for (int i = 0; i < {{ndim}}; ++i) {
		sol[i] = rhs[i];
		for (int j = 0; j < i; ++j) {
			sol[i] -= mat[i*{{ndim}}+j] * mat[j*{{ndim}}+j] * sol[j];
		}
		sol[i] /= mat[i*{{ndim}}+i];
	}
}
""")

	type: str = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'forward_subs_unit_diaged')

	funcid = forward_subs_unit_diaged_funcid(ndim, dtype)

	return codestr.render(funcid=funcid, fp_type=type, ndim=ndim)

class ForwardSubsUnitDiaged(CudaFunction):
	def __init__(self, ndim: int, dtype: cp.dtype):
		self.ndim = ndim
		self.dtype = dtype
		self.funcid = forward_subs_unit_diaged_funcid(ndim, dtype)
		self.code = forward_subs_unit_diaged_code(ndim, dtype)
		self.type_str = ctc.type_to_typestr(dtype)

	def get_device_funcid(self):
		return self.funcid

	def get_device_code(self):
		return self.code

	def get_deps(self):
		return list()


def backward_subs_unit_t_funcid(ndim: int, dtype: cp.dtype):
	return 'backward_subs_unit_t' + ctc.dim_type_funcid(ndim, dtype)

def backward_subs_unit_t_code(ndim: int, dtype: cp.dtype):
	codestr = Template(
"""
__device__
void {{funcid}}(const {{fp_type}}* mat, const {{fp_type}}* rhs, {{fp_type}}* sol) {
	#pragma unroll
	for (int i = {{ndim}} - 1; i >= 0; --i) {
		sol[i] = rhs[i];
		for (int j = i + 1; j < {{ndim}}; ++j) {
			sol[i] -= mat[j*{{ndim}}+i] * sol[j];
		}
	}
}
""")

	type: str = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'backward_subs_unit_t')

	funcid = backward_subs_unit_t_funcid(ndim, dtype)

	return codestr.render(funcid=funcid, fp_type=type, ndim=ndim)

class BackwardSubsUnitT(CudaFunction):
	def __init__(self, ndim: int, dtype: cp.dtype):
		self.ndim = ndim
		self.dtype = dtype
		self.funcid = backward_subs_unit_t_funcid(ndim, dtype)
		self.code = backward_subs_unit_t_code(ndim, dtype)
		self.type_str = ctc.type_to_typestr(dtype)

	def get_device_funcid(self):
		return self.funcid

	def get_device_code(self):
		return self.code

	def get_deps(self):
		return list()


# SOLVERS
# solves (L @ D @ L^T) @ x = y
def ldl_solve_funcid(ndim: int, dtype: cp.dtype):
	return 'ldl_solve' + ctc.dim_type_funcid(ndim, dtype)

def ldl_solve_code(ndim: int, dtype: cp.dtype):
	codestr = Template(
"""
__device__
void {{funcid}}(const {{fp_type}}* mat, const {{fp_type}}* rhs, {{fp_type}}* sol) {
	{{fp_type}} arr[{{ndim}}];
	{{forward_funcid}}(mat, rhs, arr);
	{{backward_funcid}}(mat, arr, sol);
}
""")

	type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'ldl_solve')

	funcid = ldl_solve_funcid(ndim, dtype)
	forward_fid = forward_subs_unit_diaged_funcid(ndim, dtype)
	backward_fid = backward_subs_unit_t_funcid(ndim, dtype)

	return codestr.render(funcid=funcid, fp_type=type, ndim=ndim, 
		forward_funcid=forward_fid, backward_funcid=backward_fid)

class LDLSolve(CudaFunction):
	def __init__(self, ndim: int, dtype: cp.dtype):
		self.funcid = ldl_solve_funcid(ndim, dtype)
		self.code = ldl_solve_code(ndim, dtype)
		self.type_str = ctc.type_to_typestr(dtype)

		self.ndim = ndim
		self.dtype = dtype

	def get_device_funcid(self):
		return self.funcid

	def get_device_code(self):
		return self.code

	def get_deps(self):
		return [ForwardSubsUnitDiaged(self.ndim, self.dtype), BackwardSubsUnitT(self.ndim, self.dtype)]


# FULL SOLVING PROCEDURES

# GMW81 solving
def gmw81_solver_funcid(ndim: int, dtype: cp.dtype):
	return 'gmw81_solver' + ctc.dim_type_funcid(ndim, dtype)

def gmw81_solver_code(ndim: int, dtype: cp.dtype):
	codestr = Template(
"""
__device__
void {{funcid}}({{fp_type}}* mat, const {{fp_type}}* rhs, {{fp_type}}* sol) {
	//int perm[{{ndim}}];
	//{{fp_type}} arr1[{{ndim}}];
	//{{fp_type}} arr2[{{ndim}}];
	//{{diag_pivot_funcid}}(mat, perm);
	{{gmw81_funcid}}(mat);
	//{{permute_vec_funcid}}(rhs, perm, arr1);
	//{{ldl_solve_funcid}}(mat, arr1, arr2);
	{{ldl_solve_funcid}}(mat, rhs, sol);
	//{{inv_permute_vec_funcid}}(arr2, perm, sol);
}
""")

	type: str = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'gmw81_solver')

	funcid = gmw81_solver_funcid(ndim, dtype)
	diag_pivot_fid = permute.diag_pivot_funcid(ndim, dtype)
	gmw81_fid = gmw81_funcid(ndim, dtype)
	permv_fid = permute.permute_vec_funcid(ndim, dtype)
	ldlsol_fid = ldl_solve_funcid(ndim, dtype)
	ipermv_fid = permute.inv_permute_vec_funcid(ndim, dtype)
	

	return codestr.render(funcid=funcid, fp_type=type, ndim=ndim, diag_pivot_funcid=diag_pivot_fid,
		gmw81_funcid=gmw81_fid, permute_vec_funcid=permv_fid, ldl_solve_funcid=ldlsol_fid, inv_permute_vec_funcid=ipermv_fid)

class GMW81Solver(CudaFunction):
	def __init__(self, ndim: int, dtype: cp.dtype):
		self.funcid = gmw81_solver_funcid(ndim, dtype)
		self.code = gmw81_solver_code(ndim, dtype)
		self.type_str = ctc.type_to_typestr(dtype)
		self.ndim = ndim
		self.dtype = dtype

		self.mod = None
		self.run_func = None

	def run(self, mat, rhs, sol, N):
		if self.run_func == None:
			self.build()

		Nthreads = 32
		blockSize = math.ceil(N / Nthreads)
		self.run_func((blockSize,),(Nthreads,),(mat, rhs, sol, N))

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
void {{funcid}}({{fp_type}}* mat, const {{fp_type}}* rhs, {{fp_type}}* sol, int N) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N) {

		{{fp_type}} mat_copy[{{ndim}}*{{ndim}}];
		{{fp_type}} rhs_copy[{{ndim}}];
		{{fp_type}} sol_copy[{{ndim}}];

		for (int i = 0; i < {{ndim}}; ++i) {
			rhs_copy[i] = rhs[i*N+tid];
			sol_copy[i] = sol[i*N+tid];
		}
		int k = 0;
		for (int i = 0; i < {{ndim}}; ++i) {
			for (int j = 0; j <= i; ++j) {
				{{fp_type}} temp = mat[k*N+tid];
				mat_copy[i*{{ndim}}+j] = temp;
				if (i != j) {
					mat_copy[j*{{ndim}}+i] = temp;
				}
				++k;
			}
		}

		{{dfuncid}}(mat_copy, rhs_copy, sol_copy);

		for (int i = 0; i < {{ndim}}; ++i) {
			sol[i*N+tid] = sol_copy[i];
		}

		k = 0;
		for (int i = 0; i < {{ndim}}; ++i) {
			for (int j = 0; j <= i; ++j) {
				mat[k*N+tid] = mat_copy[i*{{ndim}}+j];
				++k;
			}
		}
	}
}
""")
		fid = self.get_kernel_funcid()
		dfid = self.get_device_funcid()

		ndim=self.ndim
		nmat=round(ndim*(ndim+1)/2)

		return temp.render(funcid=fid, dfuncid=dfid, fp_type=self.type_str, ndim=ndim, nmat=nmat)

	def get_full_code(self):
		code = self.get_device_code() + '\n'
		code += self.get_kernel_code()
		return code

	def build(self):
		cc = cudap.code_gen_walking(self, "")
		try:
			self.mod = cp.RawModule(code=cc)
			self.run_func = self.mod.get_function(self.get_kernel_funcid())
		except:
			with open("on_compile_fail.cu", "w") as f:
				f.write(cc)
			raise
		return cc

		self.run_func = self.mod.get_function(self.get_kernel_funcid())
		return cc

	def get_deps(self):
		return [permute.DiagPivot(self.ndim, self.dtype), GMW81(self.ndim, self.dtype),
			permute.PermuteVec(self.ndim, self.dtype), LDLSolve(self.ndim, self.dtype),
			permute.InvPermuteVec(self.ndim, self.dtype)]
