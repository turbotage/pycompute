
__device__
void gradient_convergence_4_f(const float* pars, const float* g, const float* f, const float* lower_bound, const float* upper_bound, char* step_type, float tol, int tid, int N) 
{
	if (step_type[tid] == 0) {
		return;
	}
	
	bool clamped = false;
	float clamped_norm = 0.0f;
	float temp1;
	float temp2;
	for (int i = 0; i < 4; ++i) {
		int iidx = i*N+tid;
		temp1 = pars[iidx];
		temp2 = g[iidx];
		temp2 = temp1 - temp2;
		float u = upper_bound[iidx];
		float l = lower_bound[iidx];
		if (temp2 > u) {
			clamped = true;
			temp2 = u;
		} else if (temp2 < l) {
			clamped = true;
			temp2 = l;
		}
		temp2 = temp1 - temp2;
		clamped_norm += temp2*temp2;
	}

	if (clamped_norm < tol*(1 + f[tid])) {
		if ((step_type[tid] & 1) || clamped) {
			step_type[tid] = 0;
		}
	}
}

extern "C" __global__
void k_gradient_convergence_4_f(const float* pars, const float* g, const float* f, const float* lower_bound, const float* upper_bound, char* step_type, float tol, int N) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N) {
		gradient_convergence_4_f(pars, g, f, lower_bound, upper_bound, step_type, tol, tid, N);
	}
}
