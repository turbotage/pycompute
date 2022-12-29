
__device__
void fghhl_4_1_21_f_f7def86f0a03(const float* params, const float* consts, const float* data, const float* lam, const char* step_type,
	float* f, float* g, float* h, float* hl, int tid, int Nprobs) 
{
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

__device__
void f_4_1_21_f_f7def86f0a03(const float* params, const float* consts, const float* data,
	float* f, int tid, int Nprobs) 
{
	float pars[4];
	for (int i = 0; i < 21; ++i) {
		pars[i] = params[i*Nprobs+tid];
	}

	float res;

	for (int i = 0; i < 21; ++i) {


		res = pars[0]*(expf(-pars[2]*consts[0*1*Nprobs+i*Nprobs+tid])*pars[1] + expf(-pars[3]*consts[0*1*Nprobs+i*Nprobs+tid])*(1 - pars[1]))-data[i*Nprobs+tid];

		f[tid] += res * res;
	}
}

__device__
int max_diag_abs_4_f(const float* mat, int offset) 
{
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
void row_interchange_i_4_f(float* mat, int ii, int jj) 
{
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
void col_interchange_i_4_f(float* mat, int ii, int jj) 
{
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
void diag_pivot_4_f(float* mat, int* perm) 
{
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
void gmw81_4_f(float* mat) 
{
	float t0;
	float t1 = 0.0f; // gamma
	float t2 = 0.0f; // nu
	float beta2 = 2e-7;
	float delta = 2e-7;

	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j <= i; ++j) {
			t0 = fabsf(mat[i*4+j]);
			if (i == j) {
				if (t0 > t1)
					t1 = t0;
			} else {
				if (t0 > t2)
					t2 = t0;
			}
		}
	}

	if (4 > 1) {
		t2 /= sqrtf(4*4 - 1);
	}

	if (beta2 < t1)
		beta2 = t1;
	if (beta2 < t2)
		beta2 = t2;
	t0 = t1 + t2;
	if (t0 > 1.0f)
		delta *= t0;
	// delta = eps*max(gamma + nu, 1)
	// beta2 = max(gamma, nu/sqrt(n^^2-1), eps)

	for (int j = 0; j < 4; ++j) { // compute column j
		
		for (int s = 0; s < j; ++s)
			mat[j*4+s] /= mat[s*4+s];
		for (int i = j + 1; i < 4; ++i) {
			t0 = mat[i*4+j];
			for (int s = 0; s < j; ++s)
				t0 -= mat[j*4+s] * mat[i*4+s];
			mat[i*4+j] = t0;
		}

		t1 = 0.0f;
		for (int i = j + 1; i < 4; ++i) {
			t0 = fabsf(mat[i*4+j]);
			if (t1 < t0)
				t1 = t0;
		}
		t1 *= t1;

		t2 = fabsf(mat[j*4+j]);
		if (t2 < delta)
			t2 = delta;
		t0 = t1 / beta2;
		if (t2 < t0)
			t2 = t0;
		mat[j*4+j] = t2;

		if (j < 4) {
			for (int i = j + 1; i < 4; ++i) {
				t0 = mat[i*4+j];
				mat[i*4+i] -= t0*t0/t2;
			}
		}

	}

}

__device__
void permute_vec_4_f(const float* vec, const int* perm, float* ovec) 
{
	for (int i = 0; i < 4; ++i) {
		ovec[i] = vec[perm[i]];
	}
}

__device__
void forward_subs_unit_diaged_4_f(const float* mat, const float* rhs, float* sol) 
{
	for (int i = 0; i < 4; ++i) {
		sol[i] = rhs[i];
		for (int j = 0; j < i; ++j) {
			sol[i] -= mat[i*4+j] * mat[j*4+j] * sol[j];
		}
		sol[i] /= mat[i*4+i];
	}
}

__device__
void backward_subs_unit_t_4_f(const float* mat, const float* rhs, float* sol) 
{
	for (int i = 4 - 1; i >= 0; --i) {
		sol[i] = rhs[i];
		for (int j = i + 1; j < 4; ++j) {
			sol[i] -= mat[j*4+i] * sol[j];
		}
	}
}

__device__
void ldl_solve_4_f(const float* mat, const float* rhs, float* sol) 
{
	float arr[4];
	forward_subs_unit_diaged_4_f(mat, rhs, arr);
	backward_subs_unit_t_4_f(mat, arr, sol);
}

__device__
void inv_permute_vec_4_f(const float* vec, const int* perm, float* ovec) 
{
	for (int i = 0; i < 4; ++i) {
		ovec[perm[i]] = vec[i];
	}
}

__device__
void gmw81_solver_4_f(float* mat, const float* rhs, float* sol) 
{	
	// Diagonal pivoting of matrix and right hand side
	int perm[4];
	float arr1[4];
	float arr2[4];
	diag_pivot_4_f(mat, perm);
	permute_vec_4_f(rhs, perm, arr1);
	
	// Diagonaly scale the matrix and rhs to improve condition number
	float scale[4];
	for (int i = 0; i < 4; ++i) {
		scale[i] = sqrtf(fabsf(mat[i*4+i]));
		arr1[i] /= scale[i];
	}
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j <= i; ++j) {
			mat[i*4+j] /= (scale[i] * scale[j]);
		}
	}

	gmw81_4_f(mat);
	ldl_solve_4_f(mat, arr1, arr2);

	// Unscale
	for (int i = 0; i < 4; ++i) {
		arr2[i] /= scale[i];
	}

	// Unpivot solution
	inv_permute_vec_4_f(arr2, perm, sol);
}

__device__
void gain_ratio_step_4_f(const float* f, const float* ftp, const float* pars_tp, const float* step,
	const float* g, const float* h, float* pars, 
	float* lam, char* step_type, float mu, float eta, float acc, float dec, int tid, int N) 
{

	float actual = 0.5f * (f[tid] - ftp[tid]);
	float predicted = 0.0f;

	int k = 0;
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j <= i; ++j) {
			float entry = h[k*N+tid] * step[i*N+tid] * step[j*N+tid];
			if (i == j) {
				predicted -= entry;
			} else {
				predicted -= 2.0f * entry;
			}
			++k;
		}
	}
	predicted *= 0.5f;

	for (int i = 0; i < 4; ++i) {
		int iidx = i*N+tid;
		predicted += step[iidx] * g[iidx];
	}

	float rho = actual / predicted;

	if ((rho > mu) && (actual > 0)) {
		for (int i = 0; i < 4; ++i) {
			if (tid == 100) {
				printf("cp ");
			}
			int iidx = i*N+tid;
			pars[iidx] = pars_tp[iidx];
		}
		if (rho > eta) {
			lam[tid] /= acc;
			step_type[tid] = 1;
		} else {
			step_type[tid] = 2;
		}
	} else {
		lam[tid] *= dec;
		step_type[tid] = 4;
	}

	if (predicted < 0) {
		lam[tid] *= dec;
		step_type[tid] |= 8;
	}

	if (tid == 100) {
		printf("actual=%f, rho=%f, predicted=%f, step_type=%d, f=%f\n", actual, rho, predicted, step_type[tid], f[tid]);
	}

}

__device__
void clamp_pars_4_f(const float* lower_bound, const float* upper_bound, float* pars, int tid, int N) 
{
	for (int i = 0; i < 4; ++i) {
		int index = i*N+tid;
		float p = pars[index];
		float u = upper_bound[index];
		float l = lower_bound[index];

		if (p > u) {
			pars[index] = u;
		} else if (p < l) {
			pars[index] = l;
		}
	}
}

__device__
void gradient_convergence_4_f(const float* pars, const float* g, const float* f, const float* lower_bound, const float* upper_bound, char* step_type, float tol, int tid, int N) 
{
	bool clamped = false;
	float clamped_norm = 0.0f;
	float temp1;
	float temp2;
	for (int i = 0; i < 4; ++i) {
		int iidx = i*N+tid;
		temp1 = pars[iidx];
		temp2 = g[iidx];
		temp2 = temp1 - temp2;
		float u = upper_bound[iidx];
		float l = lower_bound[iidx];
		if (temp2 > u) {
			clamped = true;
			temp2 = u;
		} else if (temp2 < l) {
			clamped = true;
			temp2 = l;
		}
		temp2 = temp1 - temp2;
		clamped_norm += temp2*temp2;
	}

	if (clamped_norm < tol*(1 + f[tid])) {
		if ((step_type[tid] & 1) || clamped) {
			step_type[tid] = 0;
		}
	}
}

__device__
void second_order_levenberg_marquardt_4_1_21_f_f7def86f0a03(const float* consts, const float* data, const float* lower_bound, const float* upper_bound, 
	float tol, float mu, float eta, float acc, float dec,
	float* params, float* params_tp, float* step, float* lam, char* step_type,
	float* f, float* ftp, float* g, float* h, float* hl, int tid, int Nprobs)
{
	// Set gradients and objective functions to zero
	{
		f[tid] = 0.0f;
		ftp[tid] = 0.0f;
		for (int i = 0; i < 4; ++i) {
			g[i*Nprobs+tid] = 0.0f;
		}
		for (int i = 0; i < 10; ++i) {
			h[i*Nprobs+tid] = 0.0f;
			hl[i*Nprobs+tid] = 0.0f;
		}
	}

	// Calculate gradients
	{
		fghhl_4_1_21_f_f7def86f0a03(params, consts, data, lam, f, g, h, hl, tid, Nprobs);
	}

	// Solve step
	{
		float* mat = hl;
		float* rhs = g;
		float* sol = step;

		float mat_copy[4*4];
		float rhs_copy[4];
		float sol_copy[4];

		for (int i = 0; i < 4; ++i) {
			rhs_copy[i] = rhs[i*Nprobs+tid];
			sol_copy[i] = sol[i*Nprobs+tid];
		}
		int k = 0;
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j <= i; ++j) {
				float temp = mat[k*Nprobs+tid];
				mat_copy[i*4+j] = temp;
				if (i != j) {
					mat_copy[j*4+i] = temp;
				}
				++k;
			}
		}

		gmw81_solver_4_f(mat_copy, rhs_copy, sol_copy);

		for (int i = 0; i < 4; ++i) {
			sol[i*Nprobs+tid] = sol_copy[i];
		}
	}

	// Remove NaN and Infs
	{
		for (int i = 0; i < 4; ++i) {
			int idx = i*Nprobs+tid;

			// Remove inf
			float si = step[idx];
			if (isnan(si) || isinf(si)) {
				step[idx] = 0.0f;
			}
		}
	}

	// Check convergence
	{
		gradient_convergence_4_f(params, g, f, lower_bound, upper_bound, step_type, tol, tid, Nprobs);
		if (step_type[tid] == 0) {
			return;
		}
	}

	// Subtract step from params
	{
		for (int i = 0; i < 4; ++i) {
			int idx = i*Nprobs+tid;
			params_tp[idx] = params[idx] - step[idx];
		}
	}

	// Calculate error at new params
	{
		f_4_1_21_f_f7def86f0a03(params_tp, consts, data, params, tid, Nprobs);
	}

	// Calculate gain ratio and determine step type
	{
		gain_ratio_step_4_f(f, ftp, params_tp, step, g, h, params, lam, step_type, mu, eta, acc, dec, tid, Nprobs);
	}

	// Clamp parameters to feasible region
	{
		clamp_pars_4_f(lower_bound, upper_bound, params, tid, Nprobs);
	}

}

extern "C" __global__
void k_second_order_levenberg_marquardt_4_1_21_f_f7def86f0a03(const float* consts, const float* data, const float* lower_bound, const float* upper_bound,
	float tol, float mu, float eta, float acc, float dec,
	float* params, float* params_tp, float* step, float* lam, char* step_type,
	float* f, float* ftp, float* g, float* h, float* hl, int Nprobs) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < Nprobs) {

		if (step_type[tid] == 0) {
			return;
		}

		second_order_levenberg_marquardt_4_1_21_f_f7def86f0a03(consts, data, lower_bound, upper_bound, tol, mu, eta, acc, dec, params, params_tp, step, lam, step_type, f, ftp, g, h, hl, tid, Nprobs);		
	}
}
