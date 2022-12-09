
__device__
void res_jac_grad_hes_hesl_4_1_21_f_f7def86f0a03(const float* params, const float* consts, const float* data, const float* lam,
	float* res, float* jac, float* grad, float* hes, float* hesl, int tid, int N, int Nelem) 
{
	float pars[4];
	int bucket = tid / 21;

	for (int i = 0; i < 4; ++i) {
		pars[i] = params[i*N+bucket];
	}

	float x0 = expf(-pars[2]*consts[0*Nelem+tid]);
	float x1 = x0*pars[1];
	float x2 = expf(-pars[3]*consts[0*Nelem+tid]);
	float x3 = x2*(1 - pars[1]);
	float x4 = x1 + x3;
	float x5 = x0 - x2;
	float x6 = pars[0]*consts[0*Nelem+tid];
	float x7 = x3*consts[0*Nelem+tid];
	float x8 = pars[0]*powf(consts[0*Nelem+tid], 2);


	res[tid] = x4*pars[0]-data[tid];

	jac[0*Nelem+tid] = x4;
	jac[1*Nelem+tid] = x5*pars[0];
	jac[2*Nelem+tid] = -x1*x6;
	jac[3*Nelem+tid] = -x7*pars[0];


	hes[0*Nelem+tid] = 0.0f*res[tid];
	hes[1*Nelem+tid] = x5*res[tid];
	hes[2*Nelem+tid] = 0.0f*res[tid];
	hes[3*Nelem+tid] = -x1*consts[0*Nelem+tid]*res[tid];
	hes[4*Nelem+tid] = -x0*x6*res[tid];
	hes[5*Nelem+tid] = x1*x8*res[tid];
	hes[6*Nelem+tid] = -x7*res[tid];
	hes[7*Nelem+tid] = x2*x6*res[tid];
	hes[8*Nelem+tid] = 0.0f*res[tid];
	hes[9*Nelem+tid] = x3*x8*res[tid];

		
	int k = 0;
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j <= i; ++j) {
			float jtemp = jac[i*Nelem+tid] * jac[j*Nelem+tid];
			int kidx = k*Nelem+tid;
			hes[kidx] += jtemp;
			if (i != j) {
				hesl[kidx] = hes[kidx];
			} else {
				hesl[kidx] = hes[kidx] + lam[tid]*jtemp;
			}
			++k;
		}
	}

	for (int i = 0; i < 4; ++i) {
		int iidx = i*Nelem+tid;
		grad[iidx] = jac[iidx] * res[tid];
	}
}

extern "C" __global__
void k_res_jac_grad_hes_hesl_4_1_21_f_f7def86f0a03(const float* params, const float* consts, const float* data, const float* lam,
	float* res, float* jac, float* grad, float* hes, float* hesl, int N, int Nelem) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < Nelem) {
		res_jac_grad_hes_hesl_4_1_21_f_f7def86f0a03(params, consts, data, lam, res, jac, grad, hes, hesl, tid, N, Nelem);
	}
}
