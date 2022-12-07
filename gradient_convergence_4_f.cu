
__device__
void gradient_convergence_4_f(const float* pars, const float* grad, const float* lower_bound, const float* upper_bound, char* step_type, float tol, int tid, int N) 
{
	bool clamped = false;
	float clamped_norm = 0.0f;
	float norm = 0.0f;
	float temp1;
	float temp2;
	int index;
	for (int i = 0; i < 4; ++i) {
		index = i*N+tid;
		temp1 = grad[index];
		norm += temp1*temp1;
		temp2 = pars[index];
		temp1 = temp2 - temp1;
		float u = upper_bound[index];
		float l = lower_bound[index];
		if (temp1 > u) {
			clamped = true;
			temp1 = u;
		} else if (temp1 < l) {
			clamped = true;
			temp1 = l;
		}
		temp1 = temp2 - temp1;
		clamped_norm += temp1*temp1;
	}

	if (clamped_norm < tol*(1 + norm)) {
		if (step_type[tid] & 1 || clamped) {
			step_type[tid] = 0;
		}
	}
}

extern "C" __global__
void k_gradient_convergence_4_f(const float* pars, const float* grad, const float* lower_bound, const float* upper_bound, char* step_type, float tol, int N) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N) {
		gradient_convergence_4_f(pars, grad, lower_bound, upper_bound, step_type, tol, tid, N);
	}
}
