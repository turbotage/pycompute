from jinja2 import Template
import cupy as cp
#from numba import jit

from .cuda_program import CudaFunction, CudaTensor
from .cuda_program import CudaTensorChecking as ctc


def max_mag_funcid(nrow: int, ncol: int, dtype: cp.dtype):
	return 'max_mag' + ctc.dim_dim_type_funcid(nrow, ncol, dtype)

def max_mag_code(nrow: int, ncol: int, dtype: cp.dtype):
	codestr = Template(
"""
__device__
int {{funcid}}({{fp_type}}* mat, int* max_row_idx, int* max_col_idx, {{fp_type}}* max) {
	*max_row_idx = 0;
	*max_col_idx = 0;
	*max = 0.0f;
	{{fp_type}} val;
	for (int i = 0; i < {{nrow}}; ++i) {
		for (int j = 0; j < {{ncol}}; ++j) {
			val = mat[i*{{ncol}} + j];
			if (val > max) {
				*max_row_idx = i;
				*max_col_idx = j;
				max = val;
			}
		}
	}
}
""")

	type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'max_mag')

	funcid = max_mag_funcid(nrow, ncol, dtype)

	return codestr.render(funcid=funcid, fp_type=type, nrow=nrow, ncol=ncol)

class MaxMag(CudaFunction):
	def __init__(self, mat: CudaTensor, max_row_idx: CudaTensor, max_col_idx: CudaTensor, max: CudaTensor):
		func_name = 'max_mag'
		ctc.check_integer(max_row_idx, func_name)
		ctc.check_integer(max_col_idx, func_name)
		ctc.check_fp32_or_fp64(max, func_name)
		ctc.check_fp32_or_fp64(mat, func_name)
		ctc.check_scalar(max, func_name)
		ctc.check_scalar(max_row_idx, func_name)
		ctc.check_scalar(max_col_idx, func_name)
		self.mat = mat
		self.max_row_idx = max_row_idx
		self.max_col_idx = max_col_idx
		self.max = max
	
	def __init__(self, mat: CudaTensor):
		func_name = 'max_mag'
		ctc.check_fp32_or_fp64(mat, func_name)
		self.mat = mat
		self.max_row_idx = None
		self.max_col_idx = None
		self.max = None
	
	def get_funcid(self):
		return max_mag_funcid(self.mat.shape[1], self.mat.shape[2], self.mat.dtype)

	def get_code(self):
		return max_mag_code(self.mat.shape[1], self.mat.shape[2], self.mat.dtype)

	def get_deps(self):
		return list()


def max_mag_subrow_funcid(nrow: int, ncol: int, dtype: cp.dtype):
	return 'max_mag_subrow' +  ctc.dim_dim_type_funcid(nrow, ncol, dtype)

def max_mag_subrow_code(nrow: int, ncol: int, dtype: cp.dtype):
	codestr = Template(
"""
__device__
void {{funcid}}({{fp_type}}* mat, int row, int start_col, int* max_idx, {{fp_type}}* max) {
	*max_idx = 0;
	*max = 0.0f;
	{{fp_type}} val;
	for (int i = start_col; i < {{ncol}}; ++i) {
		val = mat[{{nrow}}*{{ncol}} + i];
		if (val > max) {
			*max_idx = i;
			*max = val;
		}
	}
}
""")

	type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'max_mag_subrow')

	funcid = max_mag_subrow_funcid(nrow, ncol, dtype)

	return codestr.render(funcid=funcid, fp_type=type, nrow=nrow, ncol=ncol)

class MaxMagSubrow(CudaFunction):
	def __init__(self, mat: CudaTensor, row: CudaTensor, start_col: CudaTensor, max_idx: CudaTensor, max: CudaTensor):
		func_name = 'max_mag_subrow'
		ctc.check_integer(row, func_name)
		ctc.check_integer(start_col, func_name)
		ctc.check_integer(max_idx, func_name)
		ctc.check_fp32_or_fp64(mat, func_name)
		ctc.check_fp32_or_fp64(max, func_name)
		# scalar
		ctc.check_scalar(row, func_name)
		ctc.check_scalar(start_col, func_name)
		ctc.check_scalar(max_idx, func_name)
		ctc.check_scalar(max, func_name)

		self.mat = mat
		self.row = row
		self.start_col = start_col
		self.max_idx = max_idx
		self.max = max

	def __init__(self, mat: CudaTensor):
		func_name = 'max_mag_subrow'
		ctc.check_fp32_or_fp64(mat, func_name)
		self.mat = mat
		self.row = None
		self.start_col = None
		self.max_idx = None
		self.max = None
	
	def get_funcid(self):
		return max_mag_funcid(self.mat.shape[1], self.mat.shape[2], self.mat.dtype)

	def get_code(self):
		return max_mag_code(self.mat.shape[1], self.mat.shape[2], self.mat.dtype)

	def get_deps(self):
		return list()


def max_diag_abs_funcid(ndim: int, dtype: cp.dtype):
	return 'max_diag_abs' + ctc.dim_type_funcid(ndim, dtype)

def max_diag_abs_code(ndim: int, dtype: cp.dtype):
	codestr = Template(
"""
__device__
int {{funcid}}({{fp_type}}* mat, int offset) {
	{{fp_type}} max_abs = -1.0f;
	int max_index = 0;
	for (int i = offset; i < {{ndim}}; ++i) {
		if ({{abs_funcid}}(mat[i*{{ndim}} + i]) > max_abs) {
			max_index = i;
		}
	}
	return max_index;
}
""")

	type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'max_diag_abs')

	funcid = max_diag_abs_funcid(ndim, dtype)
	abs_funcid = 'fabsf' if dtype == cp.float32 else 'fabs'

	return codestr.render(funcid=funcid, fp_type=type, ndim=ndim, abs_funcid=abs_funcid)

class MaxDiagAbs(CudaFunction):
	def __init__(self, mat: CudaTensor, offset: CudaTensor):
		func_name = 'max_diag_abs'
		ctc.check_square_mat(mat, func_name)
		ctc.check_integer(offset, func_name)
		self.mat = mat
		self.offset = offset

	def __init__(self, mat: CudaTensor):
		func_name = 'max_diag_abs'
		ctc.check_square_mat(mat, func_name)
		self.mat = mat
		self.offset = None

	def get_funcid(self):
		return max_diag_abs_funcid(self.mat.shape[1], self.mat.dtype)

	def get_code(self):
		return max_diag_abs_code(self.mat.shape[1], self.mat.dtype)

	def get_deps(self):
		return list()


def row_interchange_i_funcid(ncol: int, dtype: cp.dtype):
	return 'row_interchange_i' + ctc.dim_type_funcid(ncol, dtype)

def row_interchange_i_code(ncol: int, dtype: cp.dtype):
	codestr = Template(
"""
__device__
void {{funcid}}({{fp_type}}* mat, int ii, int jj) {
	for (int k = 0; k < {{ncol}}; ++k) {
		{{fp_type}} temp;
		temp = mat[ii*{{ncol}} + k];
		mat[ii*{{ncol}} + k] = mat[jj*{{ncol}} + k];
		mat[jj*{{ncol}} + k] = temp;
	}
}
""")

	type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'row_interchange_i')

	funcid = row_interchange_i_funcid(ncol, dtype)

	return codestr.render(funcid=funcid, fp_type=type, ncol=ncol)

class RowInterchangeI(CudaFunction):
	def __init__(self, mat: CudaTensor, ii: CudaTensor, jj: CudaTensor):
		func_name = 'row_interchange_i'
		ctc.check_fp32_or_fp64(mat, func_name)
		ctc.check_integer(ii, func_name)
		ctc.check_integer(jj, func_name)
		self.mat = mat
		self.ii = ii
		self.jj = jj

	def __init__(self, mat: CudaTensor):
		func_name = 'row_interchange_i'
		ctc.check_fp32_or_fp64(mat, func_name)
		self.mat = mat
		self.ii = None
		self.jj = None

	def get_funcid(self):
		return row_interchange_i_funcid(self.mat.shape[2], self.mat.dtype)

	def get_code(self):
		return row_interchange_i_code(self.mat.shape[2], self.mat.dtype)

	def get_deps(self):
		return list()


def col_interchange_i_funcid(nrow: int, ncol: int, dtype: cp.dtype):
	return 'col_interchange_i' + ctc.dim_dim_type_funcid(nrow, ncol, dtype)

def col_interchange_i_code(nrow: int, ncol: int, dtype: cp.dtype):
	codestr = Template(
"""
__device__
void {{funcid}}({{fp_type}}* mat, int ii, int jj) {
	for (int k = 0; k < {{nrow}}; ++k) {
		{{fp_type}} temp;
		temp = mat[k*{{ncol}} + ii];
		mat[k*{{ncol}} + ii] = mat[k*{{ncol}} + jj];
		mat[k*{{ncol}} + jj] = temp;
	}
}
""")

	type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'col_interchange_i')

	funcid = col_interchange_i_funcid(nrow, ncol, dtype)

	return codestr.render(funcid=funcid, fp_type=type, ncol=ncol, nrow=nrow)

class ColInterchangeI(CudaFunction):
	def __init__(self, mat: CudaTensor, ii: CudaTensor, jj: CudaTensor):
		func_name = 'col_interchange_i'
		self.mat = mat
		ctc.check_fp32_or_fp64(mat, func_name)
		ctc.check_integer(ii, func_name)
		ctc.check_integer(jj, func_name)
		self.ii = ii
		self.jj = jj

	def __init__(self, mat: CudaTensor):
		func_name = 'col_interchange_i'
		self.mat = mat
		ctc.check_fp32_or_fp64(mat, func_name)
		self.ii = None
		self.jj = None

	def get_funcid(self):
		return col_interchange_i_funcid(self.mat.shape[1], self.mat.shape[2], self.mat.dtype)

	def get_code(self):
		return col_interchange_i_code(self.mat.shape[1], self.mat.shape[2], self.mat.dtype)

	def get_deps(self):
		return list()


def diag_pivot_funcid(ndim: int, dtype: cp.dtype):
	return 'diag_pivot' + ctc.dim_type_funcid(ndim, dtype)

def diag_pivot_code(ndim: int, dtype: cp.dtype):
	codestr = Template(
"""
__device__
void {{funcid}}({{fp_type}}* mat, int* perm) {
	for (int i = 0; i < {{ndim}}; ++i) {
		perm[i] = i;
	}
	for (int i = 0; i < {{ndim}}; ++i) {
		int max_abs = {{max_diag_abs_funcid}}(mat, i);
		{{row_inter_i_funcid}}(mat, i, max_abs);
		{{col_inter_i_funcid}}(mat, i, max_abs);
		int temp = perm[i];
		perm[i] = perm[max_abs];
		perm[max_abs] = temp;
	}
}
""")

	type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'diag_pivot')

	funcid = diag_pivot_funcid(ndim, dtype)
	max_diag_abs_fid = max_diag_abs_funcid(ndim, dtype)
	row_fid = row_interchange_i_funcid(ndim, dtype)
	col_fid = col_interchange_i_funcid(ndim, ndim, dtype)

	return codestr.render(funcid=funcid, fp_type=type, ndim=ndim, 
		max_diag_abs_funcid=max_diag_abs_fid, row_inter_i_funcid=row_fid, col_inter_i_funcid=col_fid)

class DiagPivot(CudaFunction):
	def __init__(self, mat: CudaTensor, perm: CudaTensor):
		func_name = 'diag_pivot'
		ctc.check_square_mat(mat, func_name)
		ctc.check_fp32_or_fp64(mat, func_name)
		ctc.check_integer(perm, func_name)
		ctc.check_is_vec(perm, func_name)
		self.mat = mat
		self.perm = perm

	def __init__(self, mat: CudaTensor):
		func_name = 'diag_pivot'
		ctc.check_square_mat(mat, func_name)
		ctc.check_fp32_or_fp64(mat, func_name)
		self.mat = mat
		self.perm = None

	def get_funcid(self):
		return diag_pivot_funcid(self.mat.shape[1], self.mat.dtype)

	def get_code(self):
		return diag_pivot_code(self.mat.shape[1], self.mat.dtype)

	def get_deps(self):
		return [MaxDiagAbs(self.mat), RowInterchangeI(self.mat), ColInterchangeI(self.mat)]


def permute_vec_funcid(ndim: int, dtype: cp.dtype):
	return 'permute_vec' + ctc.dim_type_funcid(ndim, dtype)

def permute_vec_code(ndim: int, dtype: cp.dtype):
	codestr = Template(
"""
__device__
void {{funcid}}({{fp_type}}* vec, const int* perm, {{fp_type}}* ovec) {
	for (int i = 0; i < {{ndim}}; ++i) {
		ovec[i] = vec[perm[i]];
	}
}
""")

	type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'permute_vec')

	funcid = permute_vec_funcid(ndim, dtype)

	return codestr.render(funcid=funcid, fp_type=type, ndim=ndim)

class PermuteVec(CudaFunction):
	def __init__(self, vec: CudaTensor, perm: CudaTensor, ovec: CudaTensor):
		func_name = 'permute_vec'
		ctc.check_is_vec(vec, func_name)
		ctc.check_fp32_or_fp64(vec, func_name)

		ctc.check_is_vec(perm, func_name)
		ctc.check_integer(perm, func_name)

		ctc.check_is_vec(ovec, func_name)
		ctc.check_fp32_or_fp64(ovec, func_name)

		ctc.check_is_same_shape(vec, perm)
		ctc.check_is_same_shape(perm, ovec)

		self.vec = vec
		self.perm = perm
		self.ovec = ovec

	def __init__(self, vec: CudaTensor):
		func_name = 'permute_vec'
		ctc.check_is_vec(vec, func_name)
		ctc.check_fp32_or_fp64(vec, func_name)

		self.vec = vec
		self.perm = None
		self.ovec = None

	def get_funcid(self):
		return permute_vec_funcid(max(self.vec.shape[1:]), self.vec.dtype)

	def get_code(self):
		return permute_vec_code(max(self.vec.shape[1:]), self.vec.dtype)

	def get_deps(self):
		return list()


def inv_permute_vec_funcid(ndim: int, dtype: cp.dtype):
	return 'inv_permute_vec' + ctc.dim_type_funcid(ndim, dtype)

def inv_permute_vec_code(ndim: int, dtype: cp.dtype):
	codestr = Template(
"""
__device__
void {{funcid}}({{fp_type}}* vec, const int* perm, {{fp_type}}* ovec) {
	for (int i = 0; i < {{ndim}}; ++i) {
		ovec[perm[i]] = vec[i];
	}
}
""")

	type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'inv_permute_vec')

	funcid = inv_permute_vec_funcid(ndim, dtype)

	return codestr.render(funcid=funcid, fp_type=type, ndim=ndim)

class InvPermuteVec(CudaFunction):
	def __init__(self, vec: CudaTensor, perm: CudaTensor, ovec: CudaTensor):
		func_name = 'inv_permute_vec'
		ctc.check_is_vec(vec, func_name)
		ctc.check_fp32_or_fp64(vec, func_name)

		ctc.check_is_vec(perm, func_name)
		ctc.check_integer(perm, func_name)

		ctc.check_is_vec(ovec, func_name)
		ctc.check_fp32_or_fp64(ovec, func_name)

		ctc.check_is_same_shape(vec, perm)
		ctc.check_is_same_shape(perm, ovec)

		self.vec = vec
		self.perm = perm
		self.ovec = ovec

	def __init__(self, vec: CudaTensor):
		func_name = 'inv_permute_vec'
		ctc.check_is_vec(vec, func_name)
		ctc.check_fp32_or_fp64(vec, func_name)

		self.vec = vec
		self.perm = None
		self.ovec = None

	def get_funcid(self):
		return inv_permute_vec_funcid(max(self.vec.shape[1:]), self.vec.dtype)

	def get_code(self):
		return inv_permute_vec_code(max(self.vec.shape[1:]), self.vec.dtype)

	def get_deps(self):
		return list()


