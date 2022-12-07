
__device__
bool clamp_pars_4_f(const float* lower_bound, const float* upper_bound, float* pars, int tid, int N) 
{
	bool clamped = false;
	for (int i = 0; i < 4; ++i) {
		int index = i*N+tid;
		float p = pars[index];
		float u = upper_bound[i*N+tid];
		float l = lower_bound[i*N+tid];
		if (p > u) {
			clamped = true;
			pars[index] = u;
		} else if (p < l) {
			clamped = true;
			pars[index] = l;
		}
	}

	return clamped;
}

extern "C" __global__
void k_clamp_pars_4_f(const float* lower_bound, const float* upper_bound, float* pars, int N) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N) {
		clamp_pars_4_f(lower_bound, upper_bound, pars, tid, N);
	}
}
