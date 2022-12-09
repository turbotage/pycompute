
__device__
void clamp_pars_4_f(const float* lower_bound, const float* upper_bound, float* pars, int tid, int N) 
{
	for (int i = 0; i < 4; ++i) {
		int index = i*N+tid;
		float p = pars[index];
		float u = upper_bound[index];
		float l = lower_bound[index];

		if (p > u) {
			pars[index] = u;
		} else if (p < l) {
			pars[index] = l;
		}
	}
}

extern "C" __global__
void k_clamp_pars_4_f(const float* lower_bound, const float* upper_bound, float* pars, int N) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N) {
		clamp_pars_4_f(lower_bound, upper_bound, pars, tid, N);
	}
}
