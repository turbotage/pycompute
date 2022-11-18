
from jinja2 import Template
import cupy as cp
#from numba import jit

import math

from .cuda_program import CudaFunction, CudaTensor
from .cuda_program import CudaTensorChecking as ctc

from . import permute
from . import linalg


# P @ (A + E) @ P^T = L @ D @ L^T
def gmw81_funcid(ndim: int, dtype: cp.dtype):
	return 'gmw81' + ctc.dim_type_funcid(ndim, dtype)

def gmw81_code(ndim: int, dtype: cp.dtype):
	codestr = Template(
"""
__device__
void {{funcid}}({{fp_type}}* mat, unsigned int tid, unsigned N) {
	{{fp_type}} m1 = 0.0f;
	{{fp_type}} m2 = 0.0f;
	{{fp_type}} beta2 = 0.0f;
	{{fp_type}} temp;
	{{fp_type}} arr[{{ndim}}];

	for (int i = 0; i < {{ndim}}; ++i) {
		temp = {{abs_func}}(mat[{{lid_fid}}(i,i)*N + tid]);
		if (m1 < temp) {
			m1 = temp;
		}
	}

	if (beta2 < m1) {
		beta2 = m1;
	}

	for (int i = 1; i < {{ndim}}; ++i) {
		for (int j = 0; j < i; ++j) {
			temp = {{abs_func}}(mat[{{lid_fid}}(i,j)*N + tid]);
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
		{{fp_type}} d = {{abs_type}}(mat[{{lid_fid}}(i,i)*N + tid]);

		if (d < {{machine_eps}}) {
			d = {{machine_eps}};
		}

		m2 = 0.0f;
		for (int j = i + 1; j < {{ndim}}; ++j) {
			temp = {{abs_func}}(mat[{{lid_fid}}(j,i)*N + tid]);
			if (m2 < temp) {
				m2 = temp;
			}
		}

		m2 *= m2;

		if (m2 > d * beta2) {
			d = m2 / beta2;
		}

		mat[{{lid_fid}}(i,i)*N + tid] = d;

		for (int j = i + 1; j < {{ndim}}; ++j) {
			arr[j] = mat[{{lid_fid}}(j,i)*N + tid];
			mat[{{lid_fid}}(j,i)*N + tid] /= d;
		}

		for (int j = i + 1; j < {{ndim}}; ++j) {
			for (int k = j; k < {{ndim}}; ++k) {
				mat[{{lid_fid}}(k,j)*N+tid] -= arr[j] * mat[{{lid_fid}}(k,i)*N+tid];
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
		machine_eps = '1e-6'
	else:
		abs_func = 'fabs'
		sqrt_func = 'sqrt'
		machine_eps = '1e-15'


	funcid = gmw81_funcid(ndim, dtype)
	lid_fid = permute.lid_funcid()

	return codestr.render(funcid=funcid, fp_type=type, ndim=ndim, lid_fid=lid_fid,
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
void {{funcid}}({{fp_type}}* mat, unsigned int N) 
{
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N) {
		{{dfuncid}}(mat, tid, N);
	}
}
""")
		fid = self.get_kernel_funcid()
		dfid = self.get_device_funcid()

		return temp.render(funcid=fid, dfuncid=dfid, fp_type=self.type_str)

	def get_deps(self):
		return [permute.LID()]

# FORWARD AND BACKWARD SUBSTITUTIONS #

def forward_subs_unit_diaged_funcid(ndim: int, dtype: cp.dtype):
	return 'forward_subs_unit_diaged' + ctc.dim_type_funcid(ndim, dtype)

def forward_subs_unit_diaged_code(ndim: int, dtype: cp.dtype):
	codestr = Template(
"""
__device__
void {{funcid}}(const {{fp_type}}* mat, {{fp_type}}* rhs, {{fp_type}}* sol, unsigned int tid, unsigned int N) {
	for (int i = 0; i < {{ndim}}; ++i) {
		unsigned int i_nt = i*N+tid;
		sol[i_nt] = rhs[i_nt];
		for (int j = 0; j < i; ++j) {
			sol[i_nt] -= mat[{{lid_fid}}(i,j)*N+tid] * mat[{{lid_fid}}(j,i)*N+tid] * sol[j*N+tid];
		}
		sol[i_nt] /= mat[{{lid_fid}}(i,i)*N+tid];
	}
}
""")

	type: str = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'forward_subs_unit_diaged')

	funcid = forward_subs_unit_diaged_funcid(ndim, dtype)

	return codestr.render(funcid=funcid, fp_type=type, ndim=ndim)

class ForwardSubsUnitDiaged(CudaFunction):
	def __init__(self, ndim: int, dtype: cp.dtype):
		self.funcid = forward_subs_unit_diaged_funcid(ndim, dtype)
		self.code = forward_subs_unit_diaged_code(ndim, dtype)
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
void {{funcid}}(const {{fp_type}}* mat, {{fp_type}}* rhs, {{fp_type}}* sol, unsigned int N) 
{
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N) {
		{{dfuncid}}(mat, rhs, sol, tid, N);
	}
}
""")
		fid = self.get_kernel_funcid()
		dfid = self.get_device_funcid()

		return temp.render(funcid=fid, dfuncid=dfid, fp_type=self.type_str)

	def get_deps(self):
		return [permute.LID()]


def backward_subs_unit_t_funcid(ndim: int, dtype: cp.dtype):
	return 'backward_subs_unit_t' + ctc.dim_type_funcid(ndim, dtype)

def backward_subs_unit_t_code(ndim: int, dtype: cp.dtype):
	codestr = Template(
"""
__device__
void {{funcid}}({{fp_type}}* mat, {{fp_type}}* rhs, {{fp_type}}* sol, unsigned int tid, unsigned int N) {
	for (int i = {{ndim}} - 1; i >= 0; --i) {
		unsigned int i_nt = i*N+tid;
		sol[i_nt] = rhs[i_nt];
		for (int j = i + 1; j < {{ndim}}; ++j) {
			sol[i_nt] -= mat[{{lid_fid}}(j,i)*N+tid] * sol[j*N+tid];
		}
	}
}
""")

	type: str = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'backward_subs_unit_t')

	funcid = backward_subs_unit_t_funcid(ndim, dtype)

	return codestr.render(funcid=funcid, fp_type=type, ndim=ndim)

class BackwardSubsUnitT(CudaFunction):
	def __init__(self, ndim: int, dtype: cp.dtype):
		self.funcid = backward_subs_unit_t_funcid(ndim, dtype)
		self.code = backward_subs_unit_t_code(ndim, dtype)
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
void {{funcid}}(const {{fp_type}}* mat, {{fp_type}}* rhs, {{fp_type}}* sol, unsigned int N) 
{
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N) {
		{{dfuncid}}(mat, rhs, sol, tid, N);
	}
}
""")
		fid = self.get_kernel_funcid()
		dfid = self.get_device_funcid()

		return temp.render(funcid=fid, dfuncid=dfid, fp_type=self.type_str)

	def get_deps(self):
		return [permute.LID()]


# SOLVERS
# solves (L @ D @ L^T) @ x = y
def ldl_solve_funcid(ndim: int, dtype: cp.dtype):
	return 'ldl_solve' + ctc.dim_type_funcid(ndim, dtype)

def ldl_solve_code(ndim: int, dtype: cp.dtype):
	codestr = Template(
"""
__device__
void {{funcid}}({{fp_type}}* mat, {{fp_type}}* rhs, {{fp_type}}* sol, unsigned int tid, unsigned int N) {
	{{fp_type}} arr[{{ndim}}];
	{{forward_funcid}}(mat, rhs, arr, tid, N);
	{{backward_funcid}}(mat, arr, sol, tid, N);
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

	def get_kernel_funcid(self):
		funcid = self.funcid
		return 'k_' + funcid

	def get_device_code(self):
		return self.code

	def get_kernel_code(self):
		temp = Template(
"""
extern \"C\" __global__
void {{funcid}}(const {{fp_type}}* mat, {{fp_type}}* rhs, {{fp_type}}* sol, unsigned int N) 
{
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N) {
		{{dfuncid}}(mat, rhs, sol, tid, N);
	}
}
""")
		fid = self.get_kernel_funcid()
		dfid = self.get_device_funcid()

		return temp.render(funcid=fid, dfuncid=dfid, fp_type=self.type_str)

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
void {{funcid}}({{fp_type}}* mat, {{fp_type}}* rhs, {{fp_type}}* sol, unsigned int tid, unsigned int N) {
	int perm[{{ndim}}];
	{{fp_type}} arr1[{{ndim}}];
	{{fp_type}} arr2[{{ndim}}];
	{{diag_pivot_funcid}}(mat, perm, tid, N);
	{{gmw81_funcid}}(mat, tid, N);
	{{permute_vec_funcid}}(rhs, perm, arr1, tid, N);
	{{ldl_solve_funcid}}(mat, arr1, arr2, tid, N);
	{{inv_permute_vec_funcid}}(arr2, perm, sol, tid, N);
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
void {{funcid}}(const {{fp_type}}* mat, {{fp_type}}* rhs, {{fp_type}}* sol, unsigned int N) 
{
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N) {
		{{dfuncid}}(mat, rhs, sol, tid, N);
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
		self.mod = cp.RawModule(code=self.get_full_code())
		self.run_func = self.mod.get_function(self.get_kernel_funcid())

	def get_deps(self):
		return [permute.DiagPivot(self.ndim, self.dtype), GMW81(self.ndim, self.dtype),
			permute.PermuteVec(self.ndim, self.dtype), LDLSolve(self.ndim, self.dtype),
			permute.InvPermuteVec(self.ndim, self.dtype)]
