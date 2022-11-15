extern "C" __global__

void sum_every_n_upto_m_2_7_f(float* sred, unsigned int N) 
{
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int rem = tid % 2;
	unsigned int nid = (tid - rem) * 7 / 2 + 2 * rem * 2;

	if (tid < 50) {
		printf("%d  %d  %d\n", tid, rem, nid);
	}

	if (nid < N && rem != 2) {
		sred[nid] += sred[nid+2];
	}

}