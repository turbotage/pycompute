
__device__
void fghhl_4_1_21_f_f7def86f0a03(const float* params, const float* consts, const float* data, const float* lam, const char* step_type,
	float* f, float* g, float* h, float* hl, int tid, int Npars, int Ndata) 
{
	int bucket = tid / 21;
	if (step_type[bucket] == 0) {
		return;
	}

	float pars[4];
	float res;

	float grad[4];
	float jac[4];
	float hes[10];
	float hesl[10];
	float lambda = lam[bucket];


	for (int i = 0; i < 4; ++i) {
		pars[i] = params[i*Npars+bucket];
	}

	float x0 = expf(-pars[2]*consts[0*Ndata+tid]);
	float x1 = x0*pars[1];
	float x2 = expf(-pars[3]*consts[0*Ndata+tid]);
	float x3 = x2*(1 - pars[1]);
	float x4 = x1 + x3;
	float x5 = x0 - x2;
	float x6 = pars[0]*consts[0*Ndata+tid];
	float x7 = x3*consts[0*Ndata+tid];
	float x8 = pars[0]*powf(consts[0*Ndata+tid], 2);


	res = x4*pars[0]-data[tid];

	jac[0] = x4;
	jac[1] = x5*pars[0];
	jac[2] = -x1*x6;
	jac[3] = -x7*pars[0];


	hes[0] = 0.0f*res;
	hes[1] = x5*res;
	hes[2] = 0.0f*res;
	hes[3] = -x1*consts[0*Ndata+tid]*res;
	hes[4] = -x0*x6*res;
	hes[5] = x1*x8*res;
	hes[6] = -x7*res;
	hes[7] = x2*x6*res;
	hes[8] = 0.0f*res;
	hes[9] = x3*x8*res;

		
	int k = 0;
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j <= i; ++j) {
			float jtemp = jac[i] * jac[j];
			hes[k] += jtemp;
			if (i != j) {
				hesl[k] = hes[k];
			} else {
				hesl[k] = hes[k] + (lambda*jtemp, );
			}
			++k;
		}
	}

	for (int i = 0; i < 4; ++i) {
		grad[i] = jac[i] * res;
	}

	res *= res;

	// Start summing up parts
	atomicAdd(&f[bucket], res);

	for (int i = 0; i < 4; ++i) {
		atomicAdd(&g[i*Npars+bucket], grad[i]);
	}

	for (int i = 0; i < 10; ++i) {
		int iidx = i*Npars+bucket;
		atomicAdd(&h[iidx], hes[i]);
		atomicAdd(&hl[iidx], hesl[i]);
	}

}

extern "C" __global__
void k_fghhl_4_1_21_f_f7def86f0a03(const float* params, const float* consts, const float* data, const float* lam, const char* step_type,
	float* f, float* g, float* h, float* hl, int Npars, int Ndata) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < Ndata) {
		fghhl_4_1_21_f_f7def86f0a03(params, consts, data, lam, step_type, f, g, h, hl, tid, Npars, Ndata);
	}
}
