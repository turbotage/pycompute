
from jinja2 import Template
import cupy as cp
#from numba import jit

from .cuda_program import CudaFunction, CudaTensor
from .cuda_program import CudaTensorChecking as ctc

from . import permute
from . import linalg



# DECOMPOSITIONS #
# A = L @ D @ L^T
def ldl_funcid(ndim: int, dtype: cp.dtype):
	return 'ldl' + ctc.dim_type_funcid(ndim, dtype)

def ldl_code(ndim: int, dtype: cp.dtype):
	codestr = Template(
"""
__device__
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

	type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'LDL')

	funcid = ldl_funcid(ndim, dtype)

	return codestr.render(funcid=funcid, fp_type=type, ndim=ndim)

class LDL(CudaFunction):
	def __init__(self, mat: CudaTensor):
		ctc.check_square_mat(mat, 'LDL')
		self.mat = mat

	def get_funcid(self):
		return ldl_funcid(self.mat.shape[1], self.mat.dtype)

	def get_code(self):
		return ldl_code(self.mat.shape[1], self.mat.dtype)

	def get_deps(self):
		return list()


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
		return list()

# P @ A = L @ U
def lu_funcid(ndim: int, dtype: cp.dtype):
	return 'lu' + ctc.dim_type_funcid(ndim, dtype)

def lu_code(ndim: int, dtype: cp.dtype):
	codestr = Template(
"""
__device__
void {{funcid}}({{fp_type}}* mat, int* pivot) {
	{{fp_type}} val;
	for (int k = 0; k < {{ndim}} - 1; ++k) {
		int max_row_idx;
		{{fp_type}} max_row_value;
		{{max_mag_subrow_funcid}}(mat, k, k, max_row_idx, max_row_value);
		{{row_interchange_i_funcid}}(mat, k, max_row_idx);
		pivot[k] = max_row_idx;

		val = mat[k*{{ndim}} + k];
		if (val > {{machine_eps}}) {
			for (int i = k + 1; i < {{ndim}}; ++i) {
				mat[i*{{ndim}} + k] /= val;
			}

			for (int i = k + 1; i < {{ndim}}; ++i) {
				for (int j = k + 1; j < {{ndim}}; ++j) {
					mat[i*{{ndim}} + j] -= mat[i*{{ndim}} + k] * mat[k*{{ndim}} + j];
				}
			}
		}

	}
}
""")

	type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'ldl_solve')

	funcid = lu_funcid(ndim, dtype)
	max_mag_sub_fid = permute.max_mag_subrow_funcid(ndim, ndim, dtype)
	row_inter_i_fid = permute.row_interchange_i_funcid(ndim, dtype)

	return codestr.render(funcid=funcid, fp_type=type, ndim=ndim, 
		max_mag_subrow_funcid=max_mag_sub_fid, row_interchange_i_funcid=row_inter_i_fid)

class LU(CudaFunction):
	def __init__(self, mat: CudaTensor):
		ctc.check_square_mat(mat, 'lu')
		self.mat = mat

	def get_funcid(self):
		return lu_funcid(self.mat.shape[1], self.mat.dtype)

	def get_code(self):
		return lu_code(self.mat.shape[1], self.mat.dtype)

	def get_deps(self):
		return list()


# FORWARD AND BACKWARD SUBSTITUTIONS #

def forward_subs_unit_diaged_funcid(ndim: int, dtype: cp.dtype):
	return 'forward_subs_unit_diaged' + ctc.dim_type_funcid(ndim, dtype)

def forward_subs_unit_diaged_code(ndim: int, dtype: cp.dtype):
	codestr = Template(
"""
__device__
void {{funcid}}({{fp_type}}* mat, {{fp_type}}* rhs, {{fp_type}}* sol) {
	for (int i = 0; i < {{ndim}}; ++i) {
		sol[i] = rhs[i];
		for (int j = 0; j < i; ++j) {
			sol[i] -= mat[i*{{ndim}} + j] * mat[j*{{ndim}} + j] * sol[j];
		}
		sol[i] /= mat[i*{{ndim}} + i];
	}
}
""")

	type: str = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'forward_subs_unit_diaged')

	funcid = forward_subs_unit_diaged_funcid(ndim, dtype)

	return codestr.render(funcid=funcid, fp_type=type, ndim=ndim)

class ForwardSubsUnitDiaged(CudaFunction):
	def __init__(self, mat: CudaTensor, rhs: CudaTensor, sol: CudaTensor):
		func_name = 'forward_subs_unit_diaged'
		ctc.check_square_mat(mat, func_name)
		ctc.check_is_vec(rhs, func_name)
		ctc.check_is_vec(sol, func_name)
		ctc.check_matvec_multipliable(mat, sol, func_name)

		self.mat = mat
		self.rhs = rhs
		self.sol = sol

	def __init__(self, mat: CudaTensor):
		func_name = 'forward_subs_unit_diaged'
		ctc.check_square_mat(mat, func_name)

		self.mat = mat
		self.rhs = None
		self.sol = None

	def get_funcid(self):
		return forward_subs_unit_diaged_funcid(self.mat.shape[1], self.mat.dtype)

	def get_code(self):
		return forward_subs_unit_diaged_code(self.mat.shape[1], self.mat.dtype)

	def get_deps(self):
		return list()


def backward_subs_unit_t_funcid(ndim: int, dtype: cp.dtype):
	return 'backward_subs_unit_t' + ctc.dim_type_funcid(ndim, dtype)

def backward_subs_unit_t_code(ndim: int, dtype: cp.dtype):
	codestr = Template(
"""
__device__
void {{funcid}}({{fp_type}}* mat, {{fp_type}}* rhs, {{fp_type}}* sol) {
	for (int i = {{ndim}} - 1; i >= 0; --i) {
		sol[i] = rhs[i];
		for (int j = i + 1; j < {{ndim}}; ++j) {
			sol[i] -= mat[j*{{ndim}} + i] * sol[j];
		}
	}
}
""")

	type: str = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'backward_subs_unit_t')

	funcid = backward_subs_unit_t_funcid(ndim, dtype)

	return codestr.render(funcid=funcid, fp_type=type, ndim=ndim)

class BackwardSubsUnitT(CudaFunction):
	def __init__(self, mat: CudaTensor, rhs: CudaTensor, sol: CudaTensor):
		func_name = 'backward_subs_unit_t'
		ctc.check_square_mat(mat, func_name)
		ctc.check_is_vec(rhs, func_name)
		ctc.check_is_vec(sol, func_name)
		ctc.check_matvec_multipliable(mat, sol, func_name)

		self.mat = mat
		self.rhs = rhs
		self.sol = sol

	def __init__(self, mat: CudaTensor):
		func_name = 'backward_subs_unit_t'
		ctc.check_square_mat(mat, func_name)

		self.mat = mat
		self.rhs = None
		self.sol = None

	def get_funcid(self):
		return backward_subs_unit_t_funcid(self.mat.shape[1], self.mat.dtype)

	def get_code(self):
		return backward_subs_unit_t_code(self.mat.shape[1], self.mat.dtype)

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
void {{funcid}}({{fp_type}}* mat, {{fp_type}}* rhs, {{fp_type}}* sol) {
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
	def __init__(self, mat: CudaTensor, rhs: CudaTensor, sol: CudaTensor):
		func_name = 'ldl_solve'
		ctc.check_square_mat(mat, func_name)
		ctc.check_is_vec(rhs, func_name)
		ctc.check_is_vec(sol, func_name)
		ctc.check_matvec_multipliable(mat, sol, func_name)

		self.mat = mat
		self.rhs = rhs
		self.sol = sol

	def __init__(self, mat: CudaTensor):
		func_name = 'ldl_solve'
		ctc.check_square_mat(mat, func_name)

		self.mat = mat
		self.rhs = None
		self.sol = None

	def get_funcid(self):
		return ldl_solve_funcid(self.mat.shape[1], self.mat.dtype)

	def get_code(self):
		return ldl_solve_code(self.mat.shape[1], self.mat.dtype)

	def get_deps(self):
		return [ForwardSubsUnitDiaged(self.mat), BackwardSubsUnitT(self.mat)]


# FULL SOLVING PROCEDURES

# LDL solving
def ldl_solver_funcid(ndim: int, dtype: cp.dtype):
	return 'ldl_solver' + ctc.dim_type_funcid(ndim, dtype)

def ldl_solver_code(ndim: int, dtype: cp.dtype):
	codestr = Template(
"""
__device__
void {{funcid}}({{fp_type}}* mat, {{fp_type}}* rhs, {{fp_type}}* sol) {
	{{ldl_funcid}}(mat);
	{{ldl_solve_funcid}}(mat, rhs, sol);
}
""")

	type: str = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'gmw81_solver')

	funcid = gmw81_solver_funcid(ndim, dtype)
	ldl_fid = ldl_funcid(ndim, dtype)
	ldlsol_fid = ldl_solve_funcid(ndim, dtype)
	
	return codestr.render(funcid=funcid, fp_type=type, ndim=ndim,
		ldl_funcid=ldl_fid, ldl_solve_funcid=ldlsol_fid)

class LDLSolver(CudaFunction):
	def __init__(self, mat: CudaTensor, rhs: CudaTensor, sol: CudaTensor):
		func_name = 'ldl_solver'
		ctc.check_square_mat(mat, func_name)
		ctc.check_is_vec(rhs, func_name)
		ctc.check_is_vec(sol, func_name)
		ctc.check_matvec_multipliable(mat, sol, func_name)

		self.mat = mat
		self.rhs = rhs
		self.sol = sol

	def get_funcid(self):
		return ldl_solver_funcid(self.mat.ndim[1], self.mat.dtype)

	def get_code(self):
		return ldl_solver_code(self.mat.ndim[1], self.mat.dtype)

	def get_deps(self):
		return list()


# GMW81 solving
def gmw81_solver_funcid(ndim: int, dtype: cp.dtype):
	return 'gmw81_solver' + ctc.dim_type_funcid(ndim, dtype)

def gmw81_solver_code(ndim: int, dtype: cp.dtype):
	codestr = Template(
"""
__device__
void {{funcid}}({{fp_type}}* mat, {{fp_type}}* rhs, {{fp_type}}* sol) {
	int perm[{{ndim}}];
	{{fp_type}} arr1[{{ndim}}];
	{{fp_type}} arr2[{{ndim}}];
	{{diag_pivot_funcid}}(mat, perm);
	{{gmw81_funcid}}(mat);
	{{permute_vec_funcid}}(rhs, perm, arr1);
	{{ldl_solve_funcid}}(mat, arr1, arr2);
	{{inv_permute_vec_funcid}}(arr2, perm, sol);
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
	def __init__(self, mat: CudaTensor, rhs: CudaTensor, sol: CudaTensor):
		func_name = 'gmw81_solver'
		ctc.check_square_mat(mat, func_name)
		ctc.check_is_vec(rhs, func_name)
		ctc.check_is_vec(sol, func_name)
		ctc.check_matvec_multipliable(mat, sol, func_name)
		type = ctc.check_fp32_or_fp64(mat, func_name)
		ctc.check_fp32_or_fp64(rhs, func_name)
		ctc.check_fp32_or_fp64(sol, func_name)

		self.mat = mat
		self.rhs = rhs
		self.sol = sol

		self.ndim = mat.shape[1]
		self.type_str = type

	def get_funcid(self):
		return gmw81_solver_funcid(self.mat.shape[1], self.mat.dtype)

	def get_code(self):
		return gmw81_solver_code(self.mat.shape[1], self.mat.dtype)

	def get_deps(self):
		return [permute.DiagPivot(self.mat), GMW81(self.mat), permute.PermuteVec(self.rhs), 
			LDLSolve(self.mat), permute.InvPermuteVec(self.rhs)]

	def get_batched_kernel(self):
		kernel_code = Template(
"""
extern "C" __global__
void {{funcid}}({{fp_type}}* mat, {{fp_type}}* rhs, {{fp_type}}* sol, unsigned int N) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N) {
		int mat_id = {{ndim}} * {{ndim}} * tid;
		int vec_id = {{ndim}} * tid;
		{{gmw81_solver_funcid}}(&mat[mat_id], &rhs[vec_id], &sol[vec_id]);
	}
}
""")

		g81_solver_fit = self.get_funcid()
		fid = 'bk_' + g81_solver_fit

		return kernel_code.render(funcid=fid, fp_type=self.type_str, 
			ndim=self.ndim, gmw81_solver_funcid=g81_solver_fit)

