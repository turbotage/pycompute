
from jinja2 import Template
import cupy as cp
#from numba import jit

from .cuda_program import CudaFunction, CudaTensor
from .cuda_program import CudaTensorChecking as ctc

from . import permute

# A = L @ D @ L^T
def zero_mat_funcid(nrow: int, ncol: int, dtype: cp.dtype):
	return 'zero_mat' + ctc.dim_type_funcid(nrow, ncol, dtype, 'zero_mat')

def zero_mat_code(nrow: int, ncol: int, dtype: cp.dtype):
	codestr = Template(
"""
__device__
void {{funcid}}({{fp_type}}* mat) {
	for (int i = 0; i < {{nrow}}*{{ncol}}; ++i) {
		mat[i] = 0.0f;
	}
}
""")

	type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'zero_mat')

	funcid = zero_mat_funcid(nrow, ncol, dtype)

	return codestr.render(funcid=funcid, fp_type=type, nrow=nrow, ncol=ncol)

class ZeroMat(CudaFunction):
	def __init__(self, mat: CudaTensor):
		self.mat = mat

	def get_funcid(self):
		return zero_mat_funcid(self.mat.shape[1], self.mat.shape[2], self.mat.dtype)

	def get_code(self):
		return zero_mat_code(self.mat.shape[1], self.mat.shape[2], self.mat.dtype)

	def get_deps(self):
		return list()


def mul_transpose_diag_mat_funcid(nrow: int, ncol: int, dtype: cp.dtype):
	return 'mul_transpose_diag_mat' + ctc.dim_type_funcid(nrow, ncol, dtype, 'mul_transpose_diag_mat')

def mul_transpose_diag_mat_code(nrow: int, ncol: int, dtype: cp.dtype):
	codestr = Template(
"""
__device__
void {{funcid}}({{fp_type}}* mat, {{fp_type}}* diag, {{fp_type}}* omat) {
	{{fp_type}} entry;
	for (int i = 0; i < {{ncol}}; ++i) {
		for (int j = 0; j <= i; ++j) {
			entry = 0.0f;
			for (int k = 0; k < {{nrow}}; ++k) {
				entry += mat[k*{{ncol}}+i] * diag[k] * mat[k*{{ncol}}+j];
			}
			omat[i*{{ncol}}+j] = entry;
			if (i != j) {
				omat[j*{{ncol}}+i] = entry;
			}
		}
	}
}
""")

	type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'mul_transpose_diag_mat')

	funcid = mul_transpose_diag_mat_funcid(nrow, ncol, dtype)

	return codestr.render(funcid=funcid, fp_type=type, nrow=nrow, ncol=ncol)

class MulTransposeDiagMat(CudaFunction):
	def __init__(self, mat: CudaTensor, diag: CudaTensor):
		self.mat = mat
		self.diag = diag

	def __init__(self, mat: CudaTensor):
		self.mat = mat
		self.diag = None

	def get_funcid(self):
		return mul_transpose_diag_mat_funcid(self.mat.shape[1], self.mat.shape[2], self.mat.dtype)

	def get_code(self):
		return mul_transpose_diag_mat_code(self.mat.shape[1], self.mat.shape[2], self.mat.dtype)

	def get_deps(self):
		return list()


def add_mat_mat_ldiag_funcid(ndim: int, dtype: cp.dtype):
	return 'add_mat_mat_ldiag' + ctc.dim_type_funcid(ndim, dtype, 'add_mat_mat_ldiag')

def add_mat_mat_ldiag_code(ndim: int, dtype: cp.dtype):
	codestr = Template(
"""
__device__
void {{funcid}}({{fp_type}}* mat {{fp_type}} lambda, {{fp_type}}* lmat) {
	float entry1;
	float entry2;
	for (int i = 0; i < {{ndim}}; ++i) {
		for (int j = 0; j < {{ndim}}; ++j) {
			entry1 = mat[i*{{ndim}}+j];
			entry2 = lmat[i*{{ndim}}+j];
			mat[i*{{ndim}}+j] += entry2;
			mat[i*{{ndim}}+j] += entry1;
			if (i == j) {
				lmat[i*{{ndim}}+j] += lambda * entry2;
			}
		}
	}
}
""")

	type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'add_mat_mat_ldiag')

	funcid = add_mat_mat_ldiag_funcid(ndim, dtype)

	return codestr.render(funcid=funcid, fp_type=type, ndim=ndim)

class AddMatMatLdiag(CudaFunction):
	def __init__(self, mat: CudaTensor, lam: CudaTensor):
		ctc.check_square_mat(mat, 'add_mat_mat_ldiag')
		ctc.check_scalar(lam, 'add_mat_mat_ldiag')
		self.mat = mat
		self.lam = lam

	def __init__(self, mat: CudaTensor):
		ctc.check_square_mat(mat, 'add_mat_mat_ldiag')
		self.mat = mat
		self.diag = None

	def get_funcid(self):
		return add_mat_mat_ldiag_funcid(self.mat.shape[1], self.mat.dtype)

	def get_code(self):
		return add_mat_mat_ldiag_code(self.mat.shape[1], self.mat.dtype)

	def get_deps(self):
		return list()

