extern "C" __global__

void sum_every_n_upto_m_1_7_f(float* sred, unsigned int N) 
{
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int rem = tid % 4;
	unsigned int nid = (tid - rem) * 7 / 4 + 2 * rem * 1;

	if (nid < N && rem != 4) {
		sred[nid] += sred[nid+1];
	}

}