
__device__
int max_diag_abs_4_f(const float* mat, int offset) {
	float max_abs = -1.0f;
	int max_index = 0;
	for (int i = offset; i < 4; ++i) {
		if (fabsf(mat[i*4+i]) > max_abs) {
			max_index = i;
		}
	}
	return max_index;
}

__device__
void row_interchange_i_4_f(float* mat, int ii, int jj) {
	for (int k = 0; k < 4; ++k) {
		int ikn = ii*4+k;
		int jkn = jj*4+k;

		float temp;
		temp = mat[ikn];
		mat[ikn] = mat[jkn];
		mat[jkn] = temp;
	}
}

__device__
void col_interchange_i_4_f(float* mat, int ii, int jj) {
	for (int k = 0; k < 4; ++k) {
		int kin = k*4+ii;
		int kjn = k*4+jj;

		float temp;
		temp = mat[kin];
		mat[kin] = mat[kjn];
		mat[kjn] = temp;
	}
}


__device__
void diag_pivot_4_f(float* mat, int* perm) {
	for (int i = 0; i < 4; ++i) {
		perm[i] = i;
	}
	for (int i = 0; i < 4; ++i) {
		int max_abs = max_diag_abs_4_f(mat, i);
		row_interchange_i_4_f(mat, i, max_abs);
		col_interchange_i_4_f(mat, i, max_abs);
		int temp = perm[i];
		perm[i] = perm[max_abs];
		perm[max_abs] = temp;
	}
}

__device__
void gmw81_4_f(float* mat) {
	float t0;
	float t1 = 0.0f; // gamma
	float t2 = 0.0f; // nu
	float beta2 = 5e-7;
	float delta = 5e-7;
	float arr[4];

	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j <= i; ++j) {
			t0 = fabsf(mat[i*4+i]);
			if (i == j) {
				if (t0 > t1) {
					t1 = t0;
				}
			} else {
				if (t0 > t2) {
					t2 = t0;
				}
			}
		}
	}

	if (4 > 1) {
		t2 /= sqrtf(4*4 - 1);
	}


	if (beta2 < t1) {
		beta2 = t1;
	}
	if (beta2 < t2) {
		beta2 = t2;
	}
	t0 = t1 + t2;
	if (t0 > 1.0) {
		delta *= t0;
	}
	// delta = eps*max(gamma + nu, 1)
	// beta2 = max(gamma, nu/sqrt(n^^2-1), eps)

	for (int i = 0; i < 4; ++i) {
		float d = (mat[i*4+i]);

		t1 = 0.0f;
		for (int j = i + 1; j < 4; ++j) {
			t0 = fabsf(mat[j*4+i]);
			if (t1 < temp) {
				t1 = temp;
			}
		}
		t1 *= t1; // t1 holds theta

		t0 = t1 / beta2;
		if (d < t0) {
			d = t0;
		}
		if (d < delta) {
			d = delta;
		}

		mat[i*4+i] = d;

		for (int j = i + 1; j < 4; ++j) {
			int ji = j*4+i;
			arr[j] = mat[ji];
			mat[ji] /= d;
		}

		for (int j = i + 1; j < 4; ++j) {
			for (int k = j; k < 4; ++k) {
				mat[k*4+j] -= arr[j] * mat[k*4+i];
			}
		}

	}

}

__device__
void permute_vec_4_f(const float* vec, const int* perm, float* ovec) {
	for (int i = 0; i < 4; ++i) {
		ovec[i] = vec[perm[i]];
	}
}

__device__
void forward_subs_unit_diaged_4_f(const float* mat, const float* rhs, float* sol) {
	for (int i = 0; i < 4; ++i) {
		sol[i] = rhs[i];
		for (int j = 0; j < i; ++j) {
			sol[i] -= mat[i*4+j] * mat[j*4+i] * sol[j];
		}
		sol[i] /= mat[i*4+i];
	}
}

__device__
void backward_subs_unit_t_4_f(const float* mat, const float* rhs, float* sol) {
	#pragma unroll
	for (int i = 4 - 1; i >= 0; --i) {
		sol[i] = rhs[i];
		for (int j = i + 1; j < 4; ++j) {
			sol[i] -= mat[j*4+i] * sol[j];
		}
	}
}

__device__
void ldl_solve_4_f(const float* mat, const float* rhs, float* sol) {
	float arr[4];
	forward_subs_unit_diaged_4_f(mat, rhs, arr);
	backward_subs_unit_t_4_f(mat, arr, sol);
}

__device__
void inv_permute_vec_4_f(const float* vec, const int* perm, float* ovec) {
	for (int i = 0; i < 4; ++i) {
		ovec[perm[i]] = vec[i];
	}
}

__device__
void gmw81_solver_4_f(float* mat, const float* rhs, float* sol) {
	//int perm[4];
	//float arr1[4];
	//float arr2[4];
	//diag_pivot_4_f(mat, perm);
	gmw81_4_f(mat);
	//permute_vec_4_f(rhs, perm, arr1);
	//ldl_solve_4_f(mat, arr1, arr2);
	ldl_solve_4_f(mat, rhs, sol);
	//inv_permute_vec_4_f(arr2, perm, sol);
}

extern "C" __global__
void k_gmw81_solver_4_f(float* mat, const float* rhs, float* sol, int N) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N) {

		float mat_copy[4*4];
		float rhs_copy[4];
		float sol_copy[4];

		for (int i = 0; i < 4; ++i) {
			rhs_copy[i] = rhs[i*N+tid];
			sol_copy[i] = sol[i*N+tid];
		}
		int k = 0;
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j <= i; ++j) {
				float temp = mat[k*N+tid];
				mat_copy[i*4+j] = temp;
				if (i != j) {
					mat_copy[j*4+i] = temp;
				}
				++k;
			}
		}

		gmw81_solver_4_f(mat_copy, rhs_copy, sol_copy);

		for (int i = 0; i < 4; ++i) {
			sol[i*N+tid] = sol_copy[i];
		}

		k = 0;
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j <= i; ++j) {
				mat[k*N+tid] = mat_copy[i*4+j];
				++k;
			}
		}
	}
}
