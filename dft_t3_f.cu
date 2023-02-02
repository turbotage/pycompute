
#include <cupy/complex.cuh>
__device__
void dft_t3_f(const float* parr, const float* warr, const complex<float>* varr, 
	complex<float>* oarr, int tid, int nx, int nf) 
{
	float px, py, pz;
	float wx, wy, wz;
	wx = warr[tid];
	wy = warr[nf + tid];
	wz = warr[2*nf + tid];

	float ip;

	complex<float> sum;
	complex<float> freq_term;

	for (int i = 0; i < nx; ++i) {
		px = parr[i];
		py = parr[nx + i];
		pz = parr[2*nx + i];

		ip = (wx * px + wy * py + wz * pz);

		freq_term.real(cos(ip));
		freq_term.imag(sin(ip));

		sum += varr[i] * freq_term;
	}

	oarr[tid] = sum;

}

extern "C" __global__
void k_dft_t3_f(const float* parr, const float* warr, const complex<float>* varr, 
	complex<float>* oarr, int nx, int nf) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < nf) {
		dft_t3_f(parr, warr, varr, oarr, tid, nx, nf);
	}
}
