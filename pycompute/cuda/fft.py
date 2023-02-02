
import cupy as cp
from jinja2 import Template

from pycompute.cuda import cuda_program as cudap
from pycompute.cuda.cuda_program import CudaFunction
from pycompute.cuda.cuda_program import CudaTensorChecking as ctc

import math


def dft_t3_funcid(dtype: cp.dtype, sign_positive: bool):
	sign_str = '' if sign_positive else '_neg'
	return 'dft_t3' + ctc.type_funcid(dtype, 'dft_t3') + sign_str

def dft_t3_code(dtype: cp.dtype, sign_positive: bool):
	temp = Template(
"""
#include <cupy/complex.cuh>
__device__
void {{funcid}}(const {{fp_type}}* parr, const {{fp_type}}* warr, const complex<{{fp_type}}>* varr, 
	complex<{{fp_type}}>* oarr, int tid, int nx, int nf) 
{
	float px, py, pz;
	float wx, wy, wz;
	wx = warr[tid];
	wy = warr[nf + tid];
	wz = warr[2*nf + tid];

	float ip;

	complex<{{fp_type}}> sum;
	complex<{{fp_type}}> freq_term;

	for (int i = 0; i < nx; ++i) {
		px = parr[i];
		py = parr[nx + i];
		pz = parr[2*nx + i];

		ip = {{freq_sign}}(wx * px + wy * py + wz * pz);

		freq_term.real(cos(ip));
		freq_term.imag(sin(ip));

		sum += varr[i] * freq_term;
	}

	oarr[tid] = sum;

}
""")

	type = ctc.type_to_typestr(dtype)
	funcid = dft_t3_funcid(dtype, sign_positive)
	sign_str = '' if sign_positive else '-'
	return temp.render(funcid=funcid, fp_type=type, freq_sign=sign_str)

class DftT3(CudaFunction):
	def __init__(self, dtype: cp.dtype, sign_positive: bool = True, write_to_file: bool = False):
		self.funcid = dft_t3_funcid(dtype, sign_positive)
		self.code = dft_t3_code(dtype, sign_positive)

		self.write_to_file = write_to_file

		self.type_str = ctc.type_to_typestr(dtype)
		self.dtype = dtype

		self.mod = None
		self.run_func = None

	def run(self, p, w, v, o):
		if self.run_func == None:
			self.build()

		NF = w.shape[1]
		NX = p.shape[1]

		Nthreads = 32
		blockSize = math.ceil(NF / Nthreads)
		self.run_func((blockSize,),(Nthreads,),(p, w, v, o, NX, NF))

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
void {{funcid}}(const {{fp_type}}* parr, const {{fp_type}}* warr, const complex<{{fp_type}}>* varr, 
	complex<{{fp_type}}>* oarr, int nx, int nf) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < nf) {
		{{dfuncid}}(parr, warr, varr, oarr, tid, nx, nf);
	}
}
""")
		fid = self.get_kernel_funcid()
		dfid = self.get_device_funcid()

		return temp.render(funcid=fid, fp_type=self.type_str, dfuncid=dfid)

	def get_full_code(self):
		code = self.get_device_code() + '\n'
		code += self.get_kernel_code()
		return code

	def build(self):
		cc = cudap.code_gen_walking(self, "")
		if self.write_to_file:
			with open(self.get_device_funcid() + '.cu', "w") as f:
				f.write(cc)
		try:
			self.mod = cp.RawModule(code=cc)
			self.run_func = self.mod.get_function(self.get_kernel_funcid())
		except:
			with open("on_compile_fail.cu", "w") as f:
				f.write(cc)
			raise
		return cc

	def get_deps(self):
		return list()


class DftAdjT3(DftT3):
	def __init__(self, dtype: cp.dtype, sign_positive: bool = False, write_to_file: bool = False):
		super().__init__(dtype, sign_positive=sign_positive, write_to_file=write_to_file)


def dft_adj_dft_t3_funcid(dtype: cp.dtype):
	return 'dft_adj_dft_t3' + ctc.type_funcid(dtype, 'dft_adj_dft_t3')

def dft_adj_dft_t3_code(dtype: cp.dtype):
	temp = Template(
"""
#include <cupy/complex.cuh>
__device__
void {{funcid}}(const {{fp_type}}* parr, const complex<{{fp_type}}>* varr, 
	complex<{{fp_type}}>* oarr, int tid, int nx) 
{
	float px, py, pz;
	float pxt, pyt, pzt;
	pxt = parr[tid];
	pyt = parr[nx + tid];
	pzt = parr[2*nx + tid];

	float xdiff, ydiff, zdiff;
	float sinc;

	complex<{{fp_type}}> sum;

	for (int i = 0; i < nx; ++i) {
		if (i != tid) {
			px = parr[i];
			py = parr[nx + i];
			pz = parr[2*nx + i];

			xdiff = px - pxt;
			ydiff = py - pyt;
			zdiff = pz - pzt;

			sinc = (sin(xdiff) * sin(ydiff) * sin(zdiff)) / (xdiff * ydiff * zdiff);

			sum += varr[i] * sinc;
		} else {
			sum += varr[i];
		}
	}

	oarr[tid] = sum;

}
""")

	type = ctc.type_to_typestr(dtype)
	funcid = dft_adj_dft_t3_funcid(dtype)
	return temp.render(funcid=funcid, fp_type=type)

class DftAdjDftT3(CudaFunction):
	def __init__(self, dtype: cp.dtype, write_to_file: bool = False):
		self.funcid = dft_adj_dft_t3_funcid(dtype)
		self.code = dft_adj_dft_t3_code(dtype)

		self.write_to_file = write_to_file

		self.type_str = ctc.type_to_typestr(dtype)
		self.dtype = dtype

		self.mod = None
		self.run_func = None

	def run(self, p, v, o):
		if self.run_func == None:
			self.build()

		NX = p.shape[1]

		Nthreads = 32
		blockSize = math.ceil(NX / Nthreads)
		self.run_func((blockSize,),(Nthreads,),(p, v, o, NX))

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
void {{funcid}}(const {{fp_type}}* parr, const complex<{{fp_type}}>* varr, 
	complex<{{fp_type}}>* oarr, int nx) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < nx) {
		{{dfuncid}}(parr, varr, oarr, tid, nx);
	}
}
""")
		fid = self.get_kernel_funcid()
		dfid = self.get_device_funcid()

		return temp.render(funcid=fid, fp_type=self.type_str, dfuncid=dfid)

	def get_full_code(self):
		code = self.get_device_code() + '\n'
		code += self.get_kernel_code()
		return code

	def build(self):
		cc = cudap.code_gen_walking(self, "")
		if self.write_to_file:
			with open(self.get_device_funcid() + '.cu', "w") as f:
				f.write(cc)
		try:
			self.mod = cp.RawModule(code=cc)
			self.run_func = self.mod.get_function(self.get_kernel_funcid())
		except:
			with open("on_compile_fail.cu", "w") as f:
				f.write(cc)
			raise
		return cc

	def get_deps(self):
		return list()




