
__device__
int lid(int i, int j) {
	return i*(i+1) + j;
}

__device__
int max_diag_abs_4_f(const float* mat, int offset) {
	float max_abs = -1.0f;
	int max_index = 0;
	#pragma unroll
	for (int i = offset; i < 4; ++i) {
		if (fabsf(mat[lid(i,i)]) > max_abs) {
			max_index = i;
		}
	}
	return max_index;
}

__device__
void row_interchange_i_f(float* mat, int ii, int jj) {
	int ncopy = max(ii,jj);
	#pragma unroll
	for (int k = 0; k < ncopy; ++k) {
		int ikn = lid(ii,k);
		int jkn = lid(jj,k);

		float temp;
		temp = mat[ikn];
		mat[ikn] = mat[jkn];
		mat[jkn] = temp;
	}
}

__device__
void col_interchange_i_4_f(float* mat, int ii, int jj) {
	int ncopy = max(4 - ii, 4 - jj);
	#pragma unroll
	for (int k = 0; k < ncopy; ++k) {
		int kin = lid(k,ii);
		int kjn = lid(k,jj);

		float temp;
		temp = mat[kin];
		mat[kin] = mat[kjn];
		mat[kjn] = temp;
	}
}


__device__
void diag_pivot_4_f(float* mat, int* perm) {
	#pragma unroll
	for (int i = 0; i < 4; ++i) {
		perm[i] = i;
	}
	#pragma unroll
	for (int i = 0; i < 4; ++i) {
		int max_abs = max_diag_abs_4_f(mat, i);
		row_interchange_i_f(mat, i, max_abs);
		col_interchange_i_4_f(mat, i, max_abs);
		int temp = perm[i];
		perm[i] = perm[max_abs];
		perm[max_abs] = temp;
	}
}

__device__
void gmw81_4_f(float* mat, int tid, int N) {
	float m1 = 0.0f;
	float m2 = 0.0f;
	float beta2 = 0.0f;
	float temp;
	float arr[4];

	for (int i = 0; i < 4; ++i) {
		temp = fabsf(mat[lid(i,i)]);
		if (m1 < temp) {
			m1 = temp;
		}
	}

	if (beta2 < m1) {
		beta2 = m1;
	}

	for (int i = 1; i < 4; ++i) {
		for (int j = 0; j < i; ++j) {
			temp = fabsf(mat[lid(i,j)]);
			if (m2 < temp) {
				m2 = temp;
			}
		}
	}

	if (4 > 1) {
		m2 /= sqrtf(4*4 - 1);
	}

	if (beta2 < m2) {
		beta2 = m2;
	}

	for (int i = 0; i < 4; ++i) {
		float d = (mat[lid(i,i)]);

		if (d < 1e-6) {
			d = 1e-6;
		}

		m2 = 0.0f;
		for (int j = i + 1; j < 4; ++j) {
			temp = fabsf(mat[lid(j,i)]);
			if (m2 < temp) {
				m2 = temp;
			}
		}

		m2 *= m2;

		if (m2 > d * beta2) {
			d = m2 / beta2;
		}

		mat[lid(i,i)] = d;

		for (int j = i + 1; j < 4; ++j) {
			arr[j] = mat[lid(j,i)];
			mat[lid(j,i)] /= d;
		}

		for (int j = i + 1; j < 4; ++j) {
			for (int k = j; k < 4; ++k) {
				mat[lid(k,j)] -= arr[j] * mat[lid(k,i)];
			}
		}

	}

}

__device__
void permute_vec_4_f(const float* vec, const int* perm, float* ovec) {
	#pragma unroll
	for (int i = 0; i < 4; ++i) {
		ovec[i] = vec[perm[i]];
	}
}

__device__
void forward_subs_unit_diaged_4_f(const float* mat, const float* rhs, float* sol) {
	for (int i = 0; i < 4; ++i) {
		sol[i] = rhs[i];
		for (int j = 0; j < i; ++j) {
			sol[i] -= mat[lid(i,j)] * mat[lid(j,i)] * sol[j];
		}
		sol[i] /= mat[lid(i,i)];
	}
}

__device__
void backward_subs_unit_t_4_f(float* mat, const float* rhs, float* sol) {
	#pragma unroll
	for (int i = 4 - 1; i >= 0; --i) {
		sol[i] = rhs[i];
		for (int j = i + 1; j < 4; ++j) {
			sol[i] -= mat[lid(j,i)] * sol[j];
		}
	}
}

__device__
void ldl_solve_4_f(float* mat, const float* rhs, float* sol) {
	float arr[4];
	forward_subs_unit_diaged_4_f(mat, rhs, arr);
	backward_subs_unit_t_4_f(mat, arr, sol);
}

__device__
void inv_permute_vec_4_f(const float* vec, const int* perm, float* ovec) {
	#pragma unroll
	for (int i = 0; i < 4; ++i) {
		ovec[perm[i]] = vec[i];
	}
}

__device__
void gmw81_solver_4_f(float* mat, const float* rhs, float* sol) {
	int perm[4];
	float arr1[4];
	float arr2[4];
	diag_pivot_4_f(mat, perm);
	gmw81_4_f(mat);
	permute_vec_4_f(rhs, perm, arr1);
	ldl_solve_4_f(mat, arr1, arr2);
	inv_permute_vec_4_f(arr2, perm, sol);
}

extern "C" __global__
void k_gmw81_solver_4_f(float* mat, const float* rhs, float* sol, int N) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N) {

		float mat_copy[10];
		float rhs_copy[4];
		float sol_copy[4];

		#pragma unroll
		for (int i = 0; i < 4; ++i) {
			rhs_copy[i] = rhs[i*N+tid];
			sol_copy[i] = sol[i*N+tid];
		}
		#pragma unroll
		for (int i = 0; i < 10; ++i) {
			mat_copy[i] = mat[i*N+tid];
		}

		gmw81_solver_4_f(mat_copy, rhs_copy, sol_copy);

		#pragma unroll
		for (int i = 0; i < 4; ++i) {
			sol[i*N+tid] = sol_copy[i];
		}
	}
}
