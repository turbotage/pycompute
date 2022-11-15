
from jinja2 import Template
import cupy as cp
#from numba import jit
import numpy as np
import math

from cuda.cuda_program import CudaFunction, CudaTensor
from cuda.cuda_program import CudaTensorChecking as ctc

from . import permute


def sum_every_n_upto_m_funcid(n: int, m: int, dtype: cp.dtype):
	return 'sum_every_n_upto_m' + ctc.dim_dim_type_funcid(n, m, dtype)

def sum_every_n_upto_m_code(n: int, m: int, dtype: cp.dtype):
	
	rjh_temp = Template(
"""
void {{funcid}}({{fp_type}}* sred, unsigned int N) 
{
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int rem = tid % {{num_threads_per_m}};
	unsigned int nid = (tid - rem) * {{melem}} / {{num_threads_per_m}} + 2 * rem * {{nelem}};

	if (nid < N && rem != {{num_threads_per_m}}) {
		sred[nid] += sred[nid+{{nelem}}];
	}

}
""")

	ntpm = math.ceil(m / (2*n))

	type_str = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'sum_every_n_upto_m')
	
	funcid = sum_every_n_upto_m_funcid(n, m, dtype)

	rjh_kernel = rjh_temp.render(funcid=funcid, fp_type=type_str, num_threads_per_m=ntpm,
		melem=m, nelem=n)

	return rjh_kernel

class SubEveryNUptoM(CudaFunction):
	def __init__(self, sred: CudaTensor, n: int, m: int):
		self.sred = sred

		self.funcid = sum_every_n_upto_m_funcid(n, m, sred.dtype)
		self.code = sum_every_n_upto_m_code(n, m, sred.dtype)

	def get_funcid(self):
		return self.funcid

	def get_device_code(self):
		return "__device__\n" + self.code

	def get_kernel_code(self):
		return "extern \"C\" __global__\n" + self.code

	def get_deps(self):
		return list()

