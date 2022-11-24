
__device__
void sum_res_jac_hes_hesl_4_21_f(const float* res, const float* grad, const float* hes, const float* hesl,
	float* f, float* g, float* h, float* hl, int tid, int N, int Nelem) 
{
	int bucket = tid / 21;
	atomicAdd(&f[bucket], res[tid]);
	for (int i = 0; i < 4; ++i) {
		atomicAdd(&g[i*N + bucket], grad[i*Nelem + tid]);
	}
	for (int i = 0; i < 10; ++i) {
		int i_nb = i*N + bucket;
		int i_nt = i*Nelem + tid;
		atomicAdd(&h[i_nb], hes[i_nt]);
		atomicAdd(&hl[i_nb], hesl[i_nt]);
	}
}

extern "C" __global__
void k_sum_res_jac_hes_hesl_4_21_f(const float* res, const float* grad, const float* hes, const float* hesl,
	float* f, float* g, float* h, float* hl, int N, int Nelem) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < Nelem) {
		sum_res_jac_hes_hesl_4_21_f(res, grad, hes, hesl, f, g, h, hl, tid, N, Nelem);
	}
}
