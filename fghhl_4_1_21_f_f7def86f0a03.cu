
__device__
void fghhl_4_1_21_f_f7def86f0a03(const float* params, const float* consts, const float* data, const float* lam, const char* step_type,
	float* f, float* g, float* h, float* hl, int tid, int Nprobs) 
{
	if (step_type[tid] == 0) {
		return;
	}

	float pars[4];
	for (int i = 0; i < 4; ++i) {
		pars[i] = params[i*Nprobs+tid];
	}

	float res;

	float jac[4];
	float hes[10];
	float lambda = lam[tid];

	for (int i = 0; i < 21; ++i) {
		
		float x0 = expf(-pars[2]*consts[0*1*Nprobs+i*Nprobs+tid]);
		float x1 = x0*pars[1];
		float x2 = expf(-pars[3]*consts[0*1*Nprobs+i*Nprobs+tid]);
		float x3 = x2*(1 - pars[1]);
		float x4 = x1 + x3;
		float x5 = x0 - x2;
		float x6 = pars[0]*consts[0*1*Nprobs+i*Nprobs+tid];
		float x7 = x3*consts[0*1*Nprobs+i*Nprobs+tid];
		float x8 = pars[0]*powf(consts[0*1*Nprobs+i*Nprobs+tid], 2);


		res = x4*pars[0]-data[i*Nprobs+tid];

		jac[0] = x4;
		jac[1] = x5*pars[0];
		jac[2] = -x1*x6;
		jac[3] = -x7*pars[0];


		hes[0] = 0.0f*res;
		hes[1] = x5*res;
		hes[2] = 0.0f*res;
		hes[3] = -x1*consts[0*1*Nprobs+i*Nprobs+tid]*res;
		hes[4] = -x0*x6*res;
		hes[5] = x1*x8*res;
		hes[6] = -x7*res;
		hes[7] = x2*x6*res;
		hes[8] = 0.0f*res;
		hes[9] = x3*x8*res;


		if (tid == 0) {
			printf("res[%d]=%f\n", i, res);
		}

		// sum these parts
		f[tid] += res * res;

		int l = 0;
		for (int j = 0; j < 4; ++j) {
			// this sums up hessian parts
			for (int k = 0; k <= j; ++k) {
				int lidx = l*Nprobs+tid;
				float jtemp = jac[j] * jac[k];
				float hjtemp = hes[l] + jtemp;
				h[lidx] += hjtemp;
				if (j != k) {
					hl[lidx] += hjtemp;
				} else {
					hl[lidx] += hjtemp + fmaxf(lambda*jtemp, 1e-4f);
				}
				++l;
			}

			g[j*Nprobs+tid] += jac[j] * res;
		}

	}
}

extern "C" __global__
void k_fghhl_4_1_21_f_f7def86f0a03(const float* params, const float* consts, const float* data, const float* lam, const char* step_type,
	float* f, float* g, float* h, float* hl, int Nprobs) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < Nprobs) {
		fghhl_4_1_21_f_f7def86f0a03(params, consts, data, lam, step_type, f, g, h, hl, tid, Nprobs);
	}
}
