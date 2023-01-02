
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

__device__
void random_search_4_1_21_f_f7def86f0a03(const float* consts, const float* data, const float* lower_bound, const float* upper_bound, 
	float* best_p, float* p, float* best_f, float* f, int tid, int Nprobs)
{
	
	// Calculate error at new params
	{
		f_4_1_21_f_f7def86f0a03(p, consts, data, f, tid, Nprobs);
	}

	// Check if new params is better and keep them in that case
	{
		if (f[tid] < best_f[tid]) {
			for (int i = 0; i < 4; ++i) {
				int iidx = i*Nprobs+tid;
				best_p[iidx] = p[iidx];
			}
			best_f[tid] = f[tid];
		}
	}

}

extern "C" __global__
void k_random_search_4_1_21_f_f7def86f0a03(const float* consts, const float* data, const float* lower_bound, const float* upper_bound, 
	float* best_p, float* p, float* best_f, float* f, int Nprobs) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < Nprobs) {
		random_search_4_1_21_f_f7def86f0a03(consts, data, lower_bound, upper_bound, best_p, p, best_f, f, tid, Nprobs);
	}
}
