
__device__
void f_4_1_21_f_f7def86f0a03(const float* params, const float* consts, const float* data,
	float* f, int tid, int Nprobs) 
{
	float pars[4];
	for (int i = 0; i < 21; ++i) {
		pars[i] = params[i*Nprobs+tid];
	}

	float res;

	for (int i = 0; i < 21; ++i) {


		res = pars[0]*(expf(-pars[2]*consts[0*1*Nprobs+i*Nprobs+tid])*pars[1] + expf(-pars[3]*consts[0*1*Nprobs+i*Nprobs+tid])*(1 - pars[1]))-data[i*Nprobs+tid];

		f[tid] += res * res;
	}
}

extern "C" __global__
void k_f_4_1_21_f_f7def86f0a03(const float* params, const float* consts, const float* data, const char* step_type,
	float* f, int Nprobs) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < Nprobs) {
		if (step_type[tid] == 0) {
			return;
		}

		f_4_1_21_f_f7def86f0a03(params, consts, data, f, tid, Nprobs);
	}
}
