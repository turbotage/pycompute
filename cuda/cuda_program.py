import cupy as cp

class CudaTensor:
    def __init__(self, shape: list[int], dtype: cp.dtype):
        self.shape = shape
        self.dtype = dtype

class CudaFunction:
    def __init__(self):
        self.deps: dict[str, CudaFunction] = []

    def gen_code(self):
        raise NotImplementedError()

    def get_deps(self):
        return self.deps


class CudaTensorChecking:

	@staticmethod
	def dim_type_funcid(ndim: int, dtype: cp.dtype, func_name: str):
		type_qualifier: str
		if dtype == cp.float32:
			type_qualifier = 'f'
		elif dtype == cp.float64:
			type_qualifier = 'd'
		else:
			raise RuntimeError('does only support fp32 and fp64 in ' + func_name)

		return '_' + ndim + '_' + type_qualifier

	@staticmethod
	def fp32_or_fp64(dtype: cp.dtype, func_name: str):
		type: str
		if dtype == cp.float32:
			type = 'float'
		elif dtype == cp.float64:
			type = 'double'
		else:
			raise RuntimeError(func_name + ' does only support fp32 and fp64')

		return type

	@staticmethod
	def check_square_mat(mat: CudaTensor, func_name: str):
		if len(mat.shape) != 3:
			raise RuntimeError(mat.name + ' shapes array must be 3 long in ' + func_name)
		if mat.shape[1] != mat.shape[2]:
			raise RuntimeError(mat.name + ' must be square in ' + func_name)

	@staticmethod
	def check_is_vec(vec: CudaTensor, func_name: str):
		if len(vec.shape) != 3:
			raise RuntimeError(vec.name + ' must be a vector in ' + func_name)

		if vec.shape[1] == 1 or vec.shape[2] == 1:
			return

		raise RuntimeError(vec.name + ' had no singleton dimension in ' + func_name)

	@staticmethod
	def check_is_same(t1: CudaTensor, t2: CudaTensor, func_name: str):
		if t1.shape != t2.shape:
			raise RuntimeError(t1.name + ' and ' + t2.name + ' must have equal shape in ' + func_name)

	@staticmethod
	def check_matrix_multipliable(m1: CudaTensor, m2: CudaTensor, func_name: str):
		if len(m1.shape) != 3:
			raise RuntimeError(m1.name + ' shapes array must be 3 long in ' + func_name)
		if len(m2.shape) != 3:
			raise RuntimeError(m2.name + ' shapes array must be 3 long in ' + func_name)
		if m1.shape[2] != m2.shape[1]:
			raise RuntimeError(m1.name + ' and ' + m2.name + ' must agree along inner dimension in ' + func_name)
		if m1.shape[0] != m2.shape[0]:
			raise RuntimeError(m1.name + ' and ' + m2.name + ' must have the same batch dimension')

	@staticmethod
	def check_vecmat_multipliable(v: CudaTensor, m: CudaTensor, func_name: str):
		if len(m.shape) != 3:
			raise RuntimeError(m.name + ' shapes array must be 3 long in ' + func_name)
		if v.shape[2] != m.shape[1]:
			raise RuntimeError(v.name + ' and ' + m.name + ' must agree along inner dimension in ' + func_name)
		if v.shape[0] != m.shape[0]:
			raise RuntimeError(v.name + ' and ' + m.name + ' must have the same batch dimension')

	@staticmethod
	def check_matvec_multipliable(m: CudaTensor, v: CudaTensor, func_name: str):
		if len(m.shape) != 3:
			raise RuntimeError(m.name + ' shapes array must be 3 long in ' + func_name)
		if m.shape[2] != v.shape[1]:
			raise RuntimeError(v.name + ' and ' + m.name + ' must agree along inner dimension in ' + func_name)
		if v.shape[0] != m.shape[0]:
			raise RuntimeError(v.name + ' and ' + m.name + ' must have the same batch dimension')

	