
__device__
void f_4_1_21_f_f7def86f0a03(const float* params, const float* consts, const float* data, const char* step_type,
	float* f, int tid, int Npars, int Ndata) 
{
	int bucket = tid / 21;
	if (step_type[bucket] == 0) {
		return;
	}

	float pars[4];

	float res;

	for (int i = 0; i < 4; ++i) {
		pars[i] = params[i*Npars+bucket];
	}



	res = pars[0]*(expf(-pars[2]*consts[0*Ndata+tid])*pars[1] + expf(-pars[3]*consts[0*Ndata+tid])*(1 - pars[1]))-data[tid];

	res *= res;

	atomicAdd(&f[bucket], res);
}

extern "C" __global__
void k_f_4_1_21_f_f7def86f0a03(const float* params, const float* consts, const float* data, const char* step_type,
	float* f, int Npars, int Ndata) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < Ndata) {
		f_4_1_21_f_f7def86f0a03(params, consts, data, step_type, f, tid, Npars, Ndata);
	}
}
