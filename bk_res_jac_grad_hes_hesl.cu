
__device__
void res_jac_grad_hes_hesl_4_1_f_f7def86f0a03(const float* params, const float* consts, const float* data, const float* lam,
	float* res, float* jac, float* grad, float* hes, float* hesl, unsigned int tid, unsigned int N) 
{
	float pars[4];
	#pragma unroll
	for (int i = 0; i < 4; ++i) {
		pars[i] = params[i*N+tid];
	}

	float x0 = expf(-pars[2]*consts[0*N+tid]);
	float x1 = x0*pars[1];
	float x2 = expf(-pars[3]*consts[0*N+tid]);
	float x3 = x2*(1 - pars[1]);
	float x4 = x1 + x3;
	float x5 = x0 - x2;
	float x6 = pars[0]*consts[0*N+tid];
	float x7 = x3*consts[0*N+tid];
	float x8 = pars[0]*powf(consts[0*N+tid], 2);


	res[tid] = x4*pars[0]-data[tid];

	jac[0*N+tid] = x4;
	jac[1*N+tid] = x5*pars[0];
	jac[2*N+tid] = -x1*x6;
	jac[3*N+tid] = -x7*pars[0];


	hes[0*N+tid] = 0.0f*res[tid];
	hes[1*N+tid] = x5*res[tid];
	hes[2*N+tid] = 0.0f*res[tid];
	hes[3*N+tid] = -x1*consts[0*N+tid]*res[tid];
	hes[4*N+tid] = -x0*x6*res[tid];
	hes[5*N+tid] = x1*x8*res[tid];
	hes[6*N+tid] = -x7*res[tid];
	hes[7*N+tid] = x2*x6*res[tid];
	hes[8*N+tid] = 0.0f*res[tid];
	hes[9*N+tid] = x3*x8*res[tid];

		
	int k = 0;
	#pragma unroll
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j <= i; ++j) {
			float jtemp = jac[i*N+tid] * jac[j*N+tid];
			hes[k*N+tid] += jtemp;
			if (i != j) {
				hesl[k*N+tid] = hes[k*N+tid];
			} else {
				hesl[k*N+tid] = hes[k*N+tid] + lam[tid]*jtemp;
			}
			++k;
		}
	}

	#pragma unroll
	for (int i = 0; i < 4; ++i) {
		grad[i*N+tid] = jac[i*N+tid] * res[tid];
	}
}

extern "C" __global__
void k_res_jac_grad_hes_hesl_4_1_f_f7def86f0a03(const float* params, const float* consts, const float* data, const float* lam,
	float* res, float* jac, float* grad, float* hes, float* hesl, unsigned int N) 
{
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N) {
		res_jac_grad_hes_hesl_4_1_f_f7def86f0a03(params, consts, data, lam, res, jac, grad, hes, hesl, tid, N);
	}
}
