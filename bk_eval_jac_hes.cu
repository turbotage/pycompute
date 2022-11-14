
extern "C" __global__
void eval_jac_hes_21000000_4_1_f_f7def86f0a03(const float* params, const float* consts, float* eval, 
	float* jac, float* hes, unsigned int N) 
{
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N) {

		float pars[4];
		for (int i = 0; i < 4; ++i) {
			pars[i] = params[i*21000000+tid];
		}

		float x0 = expf(-pars[2]*consts[0*21000000+tid]);
		float x1 = x0*pars[1];
		float x2 = expf(-pars[3]*consts[0*21000000+tid]);
		float x3 = x2*(1 - pars[1]);
		float x4 = x1 + x3;
		float x5 = x0 - x2;
		float x6 = pars[0]*consts[0*21000000+tid];
		float x7 = x3*consts[0*21000000+tid];
		float x8 = pars[0]*powf(consts[0*21000000+tid], 2);


		eval[tid] = x4*pars[0];

		jac[0*21000000+tid] = x4;
		jac[1*21000000+tid] = x5*pars[0];
		jac[2*21000000+tid] = -x1*x6;
		jac[3*21000000+tid] = -x7*pars[0];


		hes[1*21000000+tid] = x5;
		hes[3*21000000+tid] = -x1*consts[0*21000000+tid];
		hes[4*21000000+tid] = -x0*x6;
		hes[5*21000000+tid] = x1*x8;
		hes[6*21000000+tid] = -x7;
		hes[7*21000000+tid] = x2*x6;
		hes[9*21000000+tid] = x3*x8;


	}


}
