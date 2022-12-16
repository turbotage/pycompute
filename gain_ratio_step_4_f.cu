
__device__
void gain_ratio_step_4_f(const float* f, const float* ftp, const float* pars_tp, const float* step,
	const float* g, const float* h, float* pars, 
	float* lam, char* step_type, float mu, float eta, float acc, float dec, int tid, int N) 
{
	if (step_type[tid] == 0) {
		return;
	}

	float actual = 0.5f * (f[tid] - ftp[tid]);
	float predicted = 0.0f;

	int k = 0;
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j <= i; ++j) {
			float entry = h[k*N+tid] * step[i*N+tid] * step[j*N+tid];
			if (i == j) {
				predicted -= entry;
			} else {
				predicted -= 2.0f * entry;
			}
			++k;
		}
	}
	predicted *= 0.5f;

	for (int i = 0; i < 4; ++i) {
		int iidx = i*N+tid;
		predicted -= step[iidx] * g[iidx];
	}

	float rho = actual / predicted;

	if ((rho > mu) && (actual > 0)) {
		for (int i = 0; i < 4; ++i) {
			if (tid == 100) {
				printf("cp ");
			}
			int iidx = i*N+tid;
			pars[iidx] = pars_tp[iidx];
		}
		if (rho > eta) {
			lam[tid] /= acc;
			step_type[tid] = 1;
		} else {
			step_type[tid] = 2;
		}
	} else {
		lam[tid] *= dec;
		step_type[tid] = 4;
	}

	if (predicted < 0) {
		lam[tid] *= dec;
		step_type[tid] |= 8;
	}

	if (tid == 100) {
		printf("actual=%f, rho=%f, predicted=%f, step_type=%d, f=%f\n", actual, rho, predicted, step_type[tid], f[tid]);
	}

}

extern "C" __global__
void k_gain_ratio_step_4_f(const float* f, const float* ftp, const float* pars_tp, const float* step,
	const float* g, const float* h, float* pars, 
	float* lam, char* step_type, float mu, float eta, float acc, float dec, int N) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N) {
		gain_ratio_step_4_f(f, ftp, pars_tp, step, g, h, pars, lam, step_type, mu, eta, acc, dec, tid, N);
	}
}
