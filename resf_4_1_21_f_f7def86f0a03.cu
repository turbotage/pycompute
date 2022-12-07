
__device__
void resf_4_1_21_f_f7def86f0a03(const float* params, const float* consts, const float* data,
	float* res, float* f, int tid, int N, int Nelem) 
{
	float pars[4];
	int bucket = tid / 21;
	for (int i = 0; i < 4; ++i) {
		pars[i] = params[i*N+bucket];
	}



	float rtid = pars[0]*(expf(-pars[2]*consts[0*Nelem+tid])*pars[1] + expf(-pars[3]*consts[0*Nelem+tid])*(1 - pars[1]))-data[tid];

	res[tid] = rtid;
	atomicAdd(&f[bucket], rtid*rtid);
}

extern "C" __global__
void k_resf_4_1_21_f_f7def86f0a03(const float* params, const float* consts, const float* data,
	float* res, float* f, int N, int Nelem) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N) {
		resf_4_1_21_f_f7def86f0a03(params, consts, data, res, f, tid, N, Nelem);
	}
}
