
__device__
void zero_mat_4_4_f(float* mat) {
	for (int i = 0; i < 4*4; ++i) {
		mat[i] = 0.0f;
	}
}

__device__
void mul_transpose_mat_21_4_f(const float* mat, float* omat) {
	float entry;
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j <= i; ++j) {
			entry = 0.0f;
			for (int k = 0; k < 21; ++k) {
				entry += mat[k*4+i] * mat[k*4+j];
			}
			omat[i*4+j] = entry;
			if (i != j) {
				omat[j*4+i] = entry;
			}
		}
	}
}

__device__
void add_mat_mat_ldiag_4_f(float* mat, float lambda, float* lmat) {
	float entry1;
	float entry2;
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			entry1 = mat[i*4+j];
			entry2 = lmat[i*4+j];
			mat[i*4+j] += entry2;
			lmat[i*4+j] += entry1;
			if (i == j) {
				lmat[i*4+j] += lambda * entry2;
			}
		}
	}
}

__device__
void nlsq_res_jac_hes_lhes_21_4_1_f_f7def86f0a03(const float* params, const float* consts, 
	const float* data, float lambda, float* res, 
	float* jac, float* hes, float* lhes) 
{
	zero_mat_4_4_f(hes);

	for (int i = 0; i < 21; ++i) {
		float x0 = expf(-params[0]*consts[i*1+0]);
		float x1 = x0*params[0];
		float x2 = expf(-params[0]*consts[i*1+0]);
		float x3 = x2*(1 - params[0]);
		float x4 = x1 + x3;
		float x5 = x0 - x2;
		float x6 = params[0]*consts[i*1+0];
		float x7 = x3*consts[i*1+0];
		float x8 = params[0]*powf(consts[i*1+0], 2);


		res[i] = x4*params[0] - data[i];

		jac[i*4+0] = x4;
		jac[i*4+1] = x5*params[0];
		jac[i*4+2] = -x1*x6;
		jac[i*4+3] = -x7*params[0];


		hes[1*4+0] += x5 * res[i];
		hes[2*4+0] += -x1*consts[i*1+0] * res[i];
		hes[2*4+1] += -x0*x6 * res[i];
		hes[2*4+2] += x1*x8 * res[i];
		hes[3*4+0] += -x7 * res[i];
		hes[3*4+1] += x2*x6 * res[i];
		hes[3*4+3] += x3*x8 * res[i];

	}

	for (int i = 1; i < 4; ++i) {
		for (int j = 0; j < i; ++j) {
			hes[j*4+i] = hes[i*4 + j];
		}
	}

	mul_transpose_mat_21_4_f(jac, lhes);
	add_mat_mat_ldiag_4_f(hes, lambda, lhes);
}

extern "C" __global__
void bk_nlsq_res_jac_hes_lhes_21_4_1_f_f7def86f0a03(const float* params, const float* consts, const float* data, 
	float lambda, float* res, float* jac, float* hes, float* lhes, unsigned int N) {
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N) {
		unsigned int par_id = 4 * tid;
		unsigned int con_id = 21 * 1 * tid;
		unsigned int res_id = 21 * tid;
		unsigned int jac_id = 21 * 4 * tid;
		unsigned int hes_id = 4 * 4 * tid;
		
		nlsq_res_jac_hes_lhes_21_4_1_f_f7def86f0a03(&params[par_id], &consts[con_id], &data[res_id], lambda,
			&res[res_id], &jac[jac_id], &hes[hes_id], &lhes[hes_id]);
	}
}