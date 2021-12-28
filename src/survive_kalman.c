#include "cnkalman/survive_kalman.h"
#if !defined(__FreeBSD__) && !defined(__APPLE__)
#include <malloc.h>
#endif
#include <memory.h>
#include <cnkalman/numerical_diff.h>

#include "math.h"

typedef struct CnMat survive_kalman_gain_matrix;

#define KALMAN_LOG_LEVEL 1000

#define CN_KALMAN_VERBOSE(lvl, fmt, ...)                                                                               \
	{                                                                                                                  \
		if (k->log_level >= lvl) {                                                                                     \
			fprintf(stdout, fmt "\n", __VA_ARGS__);                                                                    \
		}                                                                                                              \
	}

void survive_kalman_set_logging_level(survive_kalman_state_t *k, int v) { k->log_level = v; }

static inline bool sane_covariance(const CnMat *P) {
#ifndef NDEBUG
	for (int i = 0; i < P->rows; i++) {
		if (cnMatrixGet(P, i, i) < 0)
			return false;
	}
#ifdef USE_EIGEN
	return cnDet(P) > -1e-10;
#endif
#endif
	return true;
}

static inline FLT mul_at_ib_a(const struct CnMat *A, const struct CnMat *B) {
	FLT rtn = 0;
	CN_CREATE_STACK_MAT(V, 1, 1);
	CN_CREATE_STACK_MAT(iB, B->rows, B->cols);
	cnInvert(B, &iB, CN_INVERT_METHOD_SVD);

	CN_CREATE_STACK_MAT(AtiB, A->cols, iB.cols);
	cnGEMM(A, &iB, 1, 0, 0, &AtiB, CN_GEMM_FLAG_A_T);
	cnGEMM(&AtiB, A, 1, 0, 0, &V, 0);

	rtn = V.data[0];
	CN_FREE_STACK_MAT(AtiB);
	CN_FREE_STACK_MAT(iB);
	CN_FREE_STACK_MAT(V);

	return rtn;
}

static inline FLT mul_at_b_a(const struct CnMat *A, const struct CnMat *B) {
	CN_CREATE_STACK_MAT(V, 1, 1);
	assert(A->cols == 1);
	CN_CREATE_STACK_MAT(AtiB, 1, B->rows);
	if (B->cols > 1) {
		cnGEMM(A, B, 1, 0, 0, &AtiB, CN_GEMM_FLAG_A_T);
		cnGEMM(&AtiB, A, 1, 0, 0, &V, 0);
	} else {
	  cnElementwiseMultiply(&AtiB, A, B);
	  V.data[0] = cnDot(&AtiB, A);
	}
	// assert(V.data[0] >= 0);
	return V.data[0];
}

static void cn_print_mat_v(const survive_kalman_state_t *k, int ll, const char *name, const CnMat *M, bool newlines) {
	if (k->log_level < ll) {
		return;
	}
	char term = newlines ? '\n' : ' ';
	if (!M) {
		fprintf(stdout, "null%c", term);
		return;
	}
	fprintf(stdout, "%8s %2d x %2d:%c", name, M->rows, M->cols, term);
	FLT scale = cn_sum(M);
	for (unsigned i = 0; i < M->rows; i++) {
		for (unsigned j = 0; j < M->cols; j++) {
			FLT v = cnMatrixGet(M, i, j);
			if (v == 0)
				fprintf(stdout, "         0,\t");
			else
				fprintf(stdout, "%+7.7e,\t", v);
		}
		if (newlines && M->cols > 1)
			fprintf(stdout, "\n");
	}
	fprintf(stdout, "\n");
}
static void cn_print_mat(survive_kalman_state_t *k, const char *name, const CnMat *M, bool newlines) {
	cn_print_mat_v(k, KALMAN_LOG_LEVEL, name, M, newlines);
}

void kalman_linear_predict(FLT t, const survive_kalman_state_t *k, const CnMat *x_t0_t0, CnMat *x_t0_t1) {
	int state_cnt = k->state_cnt;
	CN_CREATE_STACK_MAT(F, state_cnt, state_cnt);
	k->F_fn(k->user, t, &F, x_t0_t0);

	// X_k|k-1 = F * X_k-1|k-1
	cnGEMM(&F, x_t0_t0, 1, 0, 0, x_t0_t1, 0);
	CN_FREE_STACK_MAT(F);
}

void user_is_q(void *user, FLT t, const struct CnMat *x, CnMat *Q_out) {
	const CnMat *q = (const CnMat *)user;
	cnScale(Q_out, q, t);
}

CN_EXPORT_FUNCTION void survive_kalman_state_reset(survive_kalman_state_t *k) {
	k->t = 0;
	cn_set_zero(&k->P);

	k->Q_fn(k->user, 10, &k->state, &k->P);
	// printf("!!!! %e\n", cnDet(&k->P));
	cn_print_mat(k, "initial Pk_k", &k->P, true);
}

static void transition_is_identity(void * user, FLT t, struct CnMat *f_out, const struct CnMat *x0) {
	cn_eye(f_out, 0);
}
CN_EXPORT_FUNCTION void survive_kalman_meas_model_init(survive_kalman_state_t *k, const char *name,
												   survive_kalman_meas_model_t *mk, kalman_measurement_model_fn_t Hfn) {
	memset(mk, 0, sizeof(*mk));
	mk->k = k;
	mk->name = name;
	mk->Hfn = Hfn;
	mk->term_criteria = (struct term_criteria_t){.max_iterations = 0, .xtol = 1e-2, .mtol = 1e-8, .minimum_step = .05};
}

void survive_kalman_state_init(survive_kalman_state_t *k, size_t state_cnt, kalman_transition_fn_t F,
							   kalman_process_noise_fn_t q_fn, void *user, FLT *state) {
	memset(k, 0, sizeof(*k));

	k->state_cnt = (int)state_cnt;
	k->F_fn = F ? F : transition_is_identity;
	k->Q_fn = q_fn ? q_fn : user_is_q;

	k->P = cnMatCalloc(k->state_cnt, k->state_cnt);

	k->Predict_fn = kalman_linear_predict;
	k->user = user;

	if (!state) {
		k->State_is_heap = true;
		state = (FLT*)calloc(1, sizeof(FLT) * k->state_cnt);
	}

	k->state = cnMat(k->state_cnt, 1, state);
}

void survive_kalman_state_free(survive_kalman_state_t *k) {
	free(k->P.data);
	k->P.data = 0;

	if (k->State_is_heap)
		free(CN_FLT_PTR(&k->state));
	k->state.data = 0;
}

void survive_kalman_predict_covariance(FLT t, const CnMat *F, const CnMat *x, survive_kalman_state_t *k) {
	int dims = k->state_cnt;

	CnMat *Pk1_k1 = &k->P;
	cn_print_mat(k, "Pk-1_k-1", Pk1_k1, 1);
	CN_CREATE_STACK_MAT(Q, dims, dims);
	k->Q_fn(k->user, t, x, &Q);

	// k->P = F * k->P * F^T + Q
	cn_ABAt_add(Pk1_k1, F, Pk1_k1, &Q);
	// printf("!!!! %f\n", cnDet(Pk1_k1));
	// assert(cnDet(Pk1_k1) >= 0);
	if (k->log_level >= KALMAN_LOG_LEVEL) {
		CN_KALMAN_VERBOSE(110, "T: %f", t);
		cn_print_mat(k, "Q", &Q, 1);
		cn_print_mat(k, "F", F, 1);
		cn_print_mat(k, "Pk1_k-1", Pk1_k1, 1);
	}
	CN_FREE_STACK_MAT(Q);
}
static void survive_kalman_find_k(survive_kalman_state_t *k, survive_kalman_gain_matrix *K, const struct CnMat *H,
								  const CnMat *R) {
	int dims = k->state_cnt;

	const CnMat *Pk_k = &k->P;

	CN_CREATE_STACK_MAT(Pk_k1Ht, dims, H->rows);

	// Pk_k1Ht = P_k|k-1 * H^T
	cnGEMM(Pk_k, H, 1, 0, 0, &Pk_k1Ht, CN_GEMM_FLAG_B_T);
	CN_CREATE_STACK_MAT(S, H->rows, H->rows);

	cn_print_mat(k, "H", H, 1);
	cn_print_mat(k, "R", R, 1);

	// S = H * P_k|k-1 * H^T + R
	if (R->cols == 1) {
		cnGEMM(H, &Pk_k1Ht, 1, 0, 0, &S, 0);
		for (int i = 0; i < S.rows; i++)
			cnMatrixSet(&S, i, i, cnMatrixGet(&S, i, i) + cn_as_const_vector(R)[i]);
	} else {
		cnGEMM(H, &Pk_k1Ht, 1, R, 1, &S, 0);
	}

	assert(cn_is_finite(&S));

	cn_print_mat(k, "Pk_k1Ht", &Pk_k1Ht, 1);
	cn_print_mat(k, "S", &S, 1);

	CN_CREATE_STACK_MAT(iS, H->rows, H->rows);
	FLT diag = 0, non_diag = 0;
#define CHECK_DIAG
#ifdef CHECK_DIAG
	for (int i = 0; i < H->rows; i++) {
		for (int j = 0; j < H->rows; j++) {
			if (i == j) {
				diag += fabs(_S[i + j * H->rows]);
				_iS[i + j * H->rows] = 1. / _S[i + j * H->rows];
			} else {
				non_diag += fabs(_S[i + j * H->rows]);
				_iS[i + j * H->rows] = 0;
			}
		}
	}
#endif
	if (diag == 0 || non_diag / diag > 1e-5) {
		cnInvert(&S, &iS, CN_INVERT_METHOD_LU);
	}
	assert(cn_is_finite(&iS));
	cn_print_mat(k, "iS", &iS, 1);

	// K = Pk_k1Ht * iS
	cnGEMM(&Pk_k1Ht, &iS, 1, 0, 0, K, 0);
	cn_print_mat(k, "K", K, 1);
}

static void survive_kalman_update_covariance(survive_kalman_state_t *k, const survive_kalman_gain_matrix *K,
											 const struct CnMat *H, const struct CnMat *R) {
	int dims = k->state_cnt;
	// Apparently cvEye isn't a thing!?
	CN_CREATE_STACK_MAT(eye, dims, dims);
	cn_set_diag_val(&eye, 1);

	CN_CREATE_STACK_MAT(ikh, dims, dims);

	// ikh = (I - K * H)
	cnGEMM(K, H, -1, &eye, 1, &ikh, 0);

	// cvGEMM does not like the same addresses for src and destination...
	CnMat *Pk_k = &k->P;
	CN_CREATE_STACK_MAT(tmp, dims, dims);
	cnCopy(Pk_k, &tmp, 0);

	CN_CREATE_STACK_MAT(kRkt, dims, dims);
	bool use_joseph_form =  R->rows == R->cols;
	if (use_joseph_form) {
	  cn_ABAt_add(&kRkt, K, R, 0);
		for (int i = 0; i < kRkt.rows; i++) {
			FLT v = cnMatrixGet(&kRkt, i, i);
			cnMatrixSet(&kRkt, i, i, v > 0 ? v : 0);
		}
		assert(sane_covariance(&kRkt));
		cn_ABAt_add(Pk_k, &ikh, &tmp, &kRkt);
	} else {
		// P_k|k = (I - K * H) * P_k|k-1
		cnGEMM(&ikh, &tmp, 1, 0, 0, Pk_k, 0);
	}
	for (int i = 0; i < k->P.rows; i++) {
		cnMatrixSet(&k->P, i, i, fabs(cnMatrixGet(&k->P, i, i)));
		for (int j = i + 1; j < k->P.rows; j++) {
			FLT v1 = cnMatrixGet(&k->P, i, j);
			FLT v2 = cnMatrixGet(&k->P, j, i);
			FLT v = (v1 + v2) / 2.;
			if (fabs(v) < 1e-10)
				v = 0;
			cnMatrixSet(&k->P, i, j, v);
			cnMatrixSet(&k->P, j, i, v);
		}
	}
	// printf("!!!? %7.7f\n", cnDet(Pk_k));
	assert(sane_covariance(Pk_k));
	assert(cn_is_finite(Pk_k));
	if (k->log_level >= KALMAN_LOG_LEVEL) {
		fprintf(stdout, "INFO gain\t");
		cn_print_mat(k, "K", K, true);

		cn_print_mat(k, "ikh", &ikh, true);

		fprintf(stdout, "INFO new Pk_k\t");
		cn_print_mat(k, "Pk_k", Pk_k, true);
	}
	CN_FREE_STACK_MAT(tmp);
	CN_FREE_STACK_MAT(ikh);
	CN_FREE_STACK_MAT(eye);
}

static inline void survive_kalman_predict(FLT t, survive_kalman_state_t *k, const CnMat *x_t0_t0, CnMat *x_t0_t1) {
	// X_k|k-1 = Predict(X_K-1|k-1)
	if (k->log_level > KALMAN_LOG_LEVEL) {
		fprintf(stdout, "INFO kalman_predict from ");
		cn_print_mat(k, "x_t0_t0", x_t0_t0, false);
	}
	assert(cn_as_const_vector(x_t0_t0) != cn_as_const_vector(x_t0_t1));
	if (t == k->t) {
		cnCopy(x_t0_t0, x_t0_t1, 0);
	} else {
		assert(t > k->t);
		k->Predict_fn(t - k->t, k, x_t0_t0, x_t0_t1);
	}
	if (k->log_level > KALMAN_LOG_LEVEL) {
		fprintf(stdout, "INFO kalman_predict to   ");
		cn_print_mat(k, "x_t0_t1", x_t0_t1, false);
	}
	if (k->datalog) {
		CN_CREATE_STACK_MAT(tmp, x_t0_t0->rows, x_t0_t1->cols);
		cn_elementwise_subtract(&tmp, x_t0_t1, x_t0_t0);
		k->datalog(k, "predict_diff", cn_as_const_vector(&tmp), tmp.rows * tmp.cols);
	}
}

typedef struct numeric_jacobian_predict_fn_ctx {
    FLT dt;
    survive_kalman_state_t *k;
} numeric_jacobian_predict_fn_ctx;
static bool numeric_jacobian_predict_fn(void * user, const struct CnMat *x, struct CnMat *y) {
    numeric_jacobian_predict_fn_ctx* ctx = user;
    ctx->k->Predict_fn(ctx->dt, ctx->k, x, y);
    return true;
}

static bool numeric_jacobian_predict(survive_kalman_state_t *k, FLT dt, const struct CnMat *x, CnMat *H) {
    numeric_jacobian_predict_fn_ctx ctx = {
            .dt = dt,
            .k = k
    };
    return cnkalman_numerical_differentiate(&ctx, cnkalman_numerical_differentiate_mode_two_sided, numeric_jacobian_predict_fn, x, H);
}

typedef struct numeric_jacobian_meas_fn_ctx {
    kalman_measurement_model_fn_t Hfn;
    void *user;
    const struct CnMat *Z;
} numeric_jacobian_meas_fn_ctx;

static bool numeric_jacobian_meas_fn(void * user, const struct CnMat *x, struct CnMat *y) {
    numeric_jacobian_meas_fn_ctx* ctx = user;
    if(ctx->Hfn(ctx->user, ctx->Z, x, y, 0) == 0)
        return false;
    // Hfn gives jacobian of measurement estimation E, y returns the residual (Z - E). So we invert it and its the form
    // we want going forward
    cnScale(y, y, -1);
    return true;
}

static bool numeric_jacobian(survive_kalman_state_t *k, kalman_measurement_model_fn_t Hfn, void *user, const struct CnMat *Z, const struct CnMat *x, CnMat *H) {
    numeric_jacobian_meas_fn_ctx ctx = {
            .Hfn = Hfn,
            .user = user,
            .Z = Z
    };
    return cnkalman_numerical_differentiate(&ctx, cnkalman_numerical_differentiate_mode_two_sided, numeric_jacobian_meas_fn, x, H);
}

static CnMat *survive_kalman_find_residual(survive_kalman_meas_model_t *mk, void *user, const struct CnMat *Z,
										   const struct CnMat *x, CnMat *y, CnMat *H) {
	survive_kalman_state_t *k = mk->k;
	kalman_measurement_model_fn_t Hfn = mk->Hfn;

	if (H) {
		cn_set_constant(H, INFINITY);
	}

	CnMat *rtn = 0;
	if (Hfn) {
        bool okay = Hfn(user, Z, x, y, H);
		if (okay == false) {
			return 0;
		}
		// k->debug_jacobian = 1;
		if (mk->debug_jacobian && H) {
			CN_CREATE_STACK_MAT(H_calc, H->rows, H->cols);

			numeric_jacobian(k, Hfn, user, Z, x, &H_calc);
			fprintf(stderr, "FJAC DEBUG BEGIN %s %d\n", mk->name, Z->rows);

			for (int j = 0; j < H->cols; j++) {
				fprintf(stderr, "FJAC PARM %d\n", j);
				for (int i = 0; i < H->rows; i++) {

					FLT deriv_u = cnMatrixGet(H, i, j);
					FLT deriv_n = cnMatrixGet(&H_calc, i, j);
					FLT diff_abs = fabs(deriv_n - deriv_u);
					FLT diff_rel = diff_abs / (deriv_n + deriv_u);

					if (diff_abs > 1e-2 && diff_rel > 1e-2) {
						fprintf(stderr, "%2d %+7.7f %+7.7f %+7.7f %+7.7f %+7.7f \n", i, cn_as_vector(y)[i], deriv_u,
								deriv_n, diff_abs, diff_rel);
					}
				}
			}
			fprintf(stderr, "FJAC DEBUG END\n");
			cn_matrix_copy(H, &H_calc);
		}
		rtn = H;
	} else {
		rtn = (struct CnMat *)user;
		cnGEMM(rtn, x, -1, Z, 1, y, 0);
	}
	assert(!rtn || cn_is_finite(rtn));

	return rtn;
}
static inline enum survive_kalman_update_extended_termination_reason
survive_kalman_termination_criteria(survive_kalman_state_t *k, const struct term_criteria_t *term_criteria,
									FLT initial_error, FLT error, FLT alpha, FLT last_error) {
	FLT minimum_step = term_criteria->minimum_step > 0 ? term_criteria->minimum_step : .01;
	if (alpha == 0 || alpha < minimum_step) {
		return survive_kalman_update_extended_termination_reason_step;
	}
	if (error == 0) {
		return survive_kalman_update_extended_termination_reason_xtol;
	}

	if (isfinite(last_error) && fabs(last_error - error) < term_criteria->xtol * error) {
		return survive_kalman_update_extended_termination_reason_xtol;
	}
	return survive_kalman_update_extended_termination_reason_none;
}

CN_EXPORT_FUNCTION FLT calculate_v(const struct CnMat *y, const struct CnMat *xDiff, const struct CnMat *iR,
							   const struct CnMat *iP, FLT *meas_part, FLT *delta_part) {
	if (delta_part == 0) {
		return *meas_part = .5 * mul_at_b_a(y, iR);
	}
	*meas_part = .5 * mul_at_b_a(y, iR);
	*delta_part = .5 * mul_at_b_a(xDiff, iP);
	return .5 * (*meas_part + *delta_part);
}

const char *survive_kalman_update_extended_termination_reason_to_str(
	enum survive_kalman_update_extended_termination_reason reason) {
	switch (reason) {
	case survive_kalman_update_extended_termination_reason_none:
		return "none";
	case survive_kalman_update_extended_termination_reason_maxiter:
		return "maxiter";
	case survive_kalman_update_extended_termination_reason_invalid_jacobian:
		return "invalid_jac";
	case survive_kalman_update_extended_termination_reason_xtol:
		return "xtol";
	case survive_kalman_update_extended_termination_reason_MAX:
		return "MAX";
	case survive_kalman_update_extended_termination_reason_step:
		return "step";
	case survive_kalman_update_extended_termination_reason_mtol:
		return "mtol";
	default:
		return "";
	}
}

// Extended Kalman Filter Modifications Based on an Optimization View Point
// https://www.diva-portal.org/smash/get/diva2:844060/FULLTEXT01.pdf
// Note that in this document, 'y' is the measurement; which we refer to as 'Z'
// throughout this code and so 'y - h(x)' from the paper is 'y' in code.
// The main driver in this document is V(x) which is defined as:
// r(x) = [ R^-.5 * y; P^-.5 * (x_t-1 * x) ]
// V(X) = r'(x) * r(x) / 2
// V(X) = 1/2 * (R^-.5 * y)' * R^-.5 * y + (P^-.5 * (x_t-1 * x))'*P^-.5 * (x_t-1 * x)
// Then owing to the identity (AB)' = B'A', and also to the fact that R and P are symmetric and so R' = R; P' = P:
// V(X) = 1/2 * (y' * (R^-.5)' * R^-.5 * y + (x_t-1 * x)' * (P^-.5)'* P^-.5 * (x_t-1 * x))
// V(X) = 1/2 * (y' * (R^-.5) * R^-.5 * y + (x_t-1 * x)' * (P^-.5)* P^-.5 * (x_t-1 * x))
// V(X) = 1/2 * (y' * (R^-1) * y + (x_t-1 * x)' * (P^-1) * (x_t-1 * x))

// Similarly, we need dV(X)/dX -- ΔV(X) --
// ΔV(X) = -[ R^-.5 * H; P^-.5]' * r(x)
// ΔV(X) = -[ R^-.5 * H; P^-.5]' * [ R^-.5 * y; P^-.5 * (x_t-1 * x) ]
// ΔV(X) =  -(R^-.5 * H) * (R^-.5 * y) - P^-.5 * (P^-.5 * (x_t-1 * x))
// ΔV(X) =  H' * R^-1 * y - P^-1 * (x_t-1 * x)
// The point of all of this is that we don't need to ever explicitly calculate R^-.5 / P^-.5; just the inverses

static FLT survive_kalman_run_iterations(survive_kalman_state_t *k, const struct CnMat *Z, const struct CnMat *R,
										 survive_kalman_meas_model_t *mk, void *user, const CnMat *x_k_k1, CnMat *K,
										 CnMat *H, CnMat *x_k_k, struct survive_kalman_update_extended_stats_t *stats) {
	int state_cnt = k->state_cnt;
	int meas_cnt = Z->rows;

	CN_CREATE_STACK_MAT(y, meas_cnt, 1);
	CN_CREATE_STACK_MAT(x_i, state_cnt, 1);
	CN_CREATE_STACK_MAT(x_i_best, state_cnt, 1);

	CN_CREATE_STACK_MAT(iR, meas_cnt, R->cols > 1 ? meas_cnt : 1);
	if (R->cols > 1) {
		cnInvert(R, &iR, CN_INVERT_METHOD_LU);
	} else {
		for (int i = 0; i < meas_cnt; i++) {
			cn_as_vector(&iR)[i] = cn_as_const_vector(R)[i] == 0 ? 0 : 1. / cn_as_const_vector(R)[i];
		}
	}
	// cn_print_mat_v(k, 100, "iR", &iR, true);

	CN_CREATE_STACK_MAT(iP, state_cnt, state_cnt);
	cnInvert(&k->P, &iP, CN_INVERT_METHOD_LU);
	assert(sane_covariance(&k->P));
	// cn_print_mat_v(k, 100, "iP", &iP, true);

	assert(cn_is_finite(&iP));
	assert(cn_is_finite(&iR));

	enum survive_kalman_update_extended_termination_reason stop_reason =
		survive_kalman_update_extended_termination_reason_none;
	FLT error = INFINITY, last_error = INFINITY;
	FLT initial_error = 0;

	cn_matrix_copy(&x_i, x_k_k1);
	int iter;
	int max_iter = mk->term_criteria.max_iterations;
	FLT meas_part, delta_part;
	CN_CREATE_STACK_MAT(Hxdiff, Z->rows, 1);
	CN_CREATE_STACK_MAT(x_update, state_cnt, 1);
	CN_CREATE_STACK_MAT(xn, x_i.rows, x_i.cols);
	CN_CREATE_STACK_MAT(x_diff, state_cnt, 1);
	CN_CREATE_STACK_MAT(iRy, meas_cnt, 1);
	CN_CREATE_STACK_MAT(iPdx, state_cnt, 1);
	CN_CREATE_STACK_MAT(dVt, state_cnt, 1);

	for (iter = 0; iter < max_iter && stop_reason == survive_kalman_update_extended_termination_reason_none; iter++) {
		// Find the residual y and possibly also the jacobian H. The user could have passed one in as 'user', or given
		// us a map function which will calculate it.
		struct CnMat *HR = survive_kalman_find_residual(mk, user, Z, &x_i, &y, H);
		if (stats) {
			stats->fevals++;
			stats->hevals++;
		}

		// If the measurement jacobian isn't calculable, the best we can do is just bail.
		if (HR == 0) {
			stop_reason = survive_kalman_update_extended_termination_reason_invalid_jacobian;
			error = INFINITY;
			break;
		}
		last_error = error;

		cnSub(&x_diff, x_k_k1, &x_i);
		error = calculate_v(&y, &x_diff, &iR, &iP, &meas_part, iter == 0 ? 0 : &delta_part);
		assert(error >= 0);

		cnGEMM(&iP, &x_diff, 1, 0, 0, &iPdx, 0);

		if (R->cols > 1) {
			cnGEMM(&iR, &y, 1, 0, 0, &iRy, 0);
		} else {
		  cnElementwiseMultiply(&iRy, &iR, &y);
		}
		cnGEMM(H, &iRy, -1, &iPdx, -1, &dVt, CN_GEMM_FLAG_A_T);

		if (iter == 0) {
			initial_error = error;
			if (stats) {
				stats->orignorm_meas += meas_part;
			}
		}

		// Run update; filling in K
		survive_kalman_find_k(k, K, H, R);

		if ((stop_reason =
				 survive_kalman_termination_criteria(k, &mk->term_criteria, initial_error, error, 1, last_error)) >
			survive_kalman_update_extended_termination_reason_none) {
			goto end_of_loop;
		}

		// x_update = K * (z - y - H * Δx)
		cnGEMM(H, &x_diff, 1, 0, 0, &Hxdiff, 0);
		cnSub(&y, &y, &Hxdiff);
		cnGEMM(K, &y, 1, &x_diff, 1, &x_update, 0);

		FLT scale = 1.;
		FLT m = cnDot(&dVt, &x_update);
		if (fabs(m) < mk->term_criteria.mtol) {
			stop_reason = survive_kalman_update_extended_termination_reason_mtol;
			break;
		}
		FLT c = .5, tau = .5;
		FLT fa = 0, fa_best = error, a_best = 0;

		FLT min_step = mk->term_criteria.minimum_step == 0 ? .05 : mk->term_criteria.minimum_step;

		bool exit_condition = false;
		while (!exit_condition) {
			exit_condition = true;
			cnAddScaled(&xn, &x_update, scale, &x_i, 1);
			mk->Hfn(user, Z, &xn, &y, 0);
			if (stats) {
				stats->fevals++;
			}

			cnSub(&x_diff, x_k_k1, &xn);

			fa = calculate_v(&y, &x_diff, &iR, &iP, &meas_part, &delta_part);
			if (k->log_level >= 1000) {
				fprintf(stdout, "%3f: %7.7f ", scale, fa);
				cn_print_mat_v(k, 1000, "at x", &xn, false);
			}
			//assert(fa >= 0);

			if (fa >= error + scale * m * c) {
				exit_condition = false;
				if (fa_best > fa) {
					fa_best = fa;
					a_best = scale;
				}
				scale = tau * scale;

				if (scale <= min_step) {
					error = fa_best;
					scale = a_best;
					break;
				}
			}
		}

		if (stats && stats->total_stats) {
			stats->total_stats->step_cnt++;
			stats->total_stats->step_acc += scale;
		}
		cnAddScaled(&x_i, &x_i, 1, &x_update, scale);

		if (k->normalize_fn) {
			k->normalize_fn(k->user, &x_i);
		}
		assert(cn_is_finite(&x_i));

	end_of_loop:
		if (k->log_level > 1000) {
			fprintf(stdout, "%3d: %7.7f / %7.7f (%f, %f, %f) ", iter, initial_error, error, scale, m,
					cnNorm(&x_update));
			cn_print_mat_v(k, 1000, "new x", &x_i, false);
		}
		if (stop_reason == survive_kalman_update_extended_termination_reason_none)
			stop_reason =
				survive_kalman_termination_criteria(k, &mk->term_criteria, initial_error, error, scale, last_error);
	}
	if (stop_reason == survive_kalman_update_extended_termination_reason_none)
		stop_reason = survive_kalman_update_extended_termination_reason_maxiter;
	bool isFailure = error > initial_error || isinf(error);
	if (stats) {
		stats->iterations = iter;
		stats->orignorm = initial_error;
		stats->bestnorm = error;
		stats->stop_reason = stop_reason;
		stats->bestnorm_meas = meas_part;
		stats->bestnorm_delta = delta_part;
		if (stats->total_stats) {
			stats->total_stats->total_runs++;
			stats->total_stats->total_failures += isFailure;
			stats->total_stats->orignorm_acc += initial_error;
			stats->total_stats->bestnorm_acc += error;
			stats->total_stats->stop_reason_counts[stop_reason]++;
			stats->total_stats->total_fevals += stats->fevals;
			stats->total_stats->total_hevals += stats->hevals;
			stats->total_stats->total_iterations += stats->iterations;
			stats->total_stats->bestnorm_meas_acc += stats->bestnorm_meas;
			stats->total_stats->bestnorm_delta_acc += stats->bestnorm_delta;
			stats->total_stats->orignorm_meas_acc += stats->orignorm_meas;
		}
	}

	if (isFailure) {
		initial_error = -1;
	} else {
		assert(cn_is_finite(H));
		assert(cn_is_finite(K));
		cn_matrix_copy(x_k_k, &x_i);
	}

	CN_FREE_STACK_MAT(Hxdiff);
	CN_FREE_STACK_MAT(x_update);
	CN_FREE_STACK_MAT(xn);
	CN_FREE_STACK_MAT(x_diff);
	CN_FREE_STACK_MAT(iRy);
	CN_FREE_STACK_MAT(iPdx);
	CN_FREE_STACK_MAT(dVt);
	CN_FREE_STACK_MAT(y);
	CN_FREE_STACK_MAT(x_i);
	CN_FREE_STACK_MAT(x_i_best);

	CN_FREE_STACK_MAT(iR);
	CN_FREE_STACK_MAT(iP);

	return initial_error;
}

void survive_kalman_predict_state(FLT t, survive_kalman_state_t *k) {
    FLT dt = t - k->t;
    assert(dt >= 0);

    int state_cnt = k->state_cnt;
    CnMat *x_k_k = &k->state;

    CN_CREATE_STACK_MAT(x_k1_k1, state_cnt, 1);
    cn_matrix_copy(&x_k1_k1, x_k_k);

    survive_kalman_predict(t, k, &x_k1_k1, x_k_k);
    if (dt > 0) {
        CN_CREATE_STACK_MAT(F, state_cnt, state_cnt);
        cn_set_constant(&F, NAN);

        k->F_fn(k->user, dt, &F, &x_k1_k1);

        if(k->debug_transition_jacobian) {
            CN_CREATE_STACK_MAT(F_calc, F.rows, F.cols);

            numeric_jacobian_predict(k, dt, &x_k1_k1, &F_calc);
            fprintf(stderr, "FJAC DEBUG BEGIN predict %d\n", x_k_k->rows);

            for (int j = 0; j < F.cols; j++) {
                fprintf(stderr, "FJAC PARM %d\n", j);
                for (int i = 0; i < F.rows; i++) {

                    FLT deriv_u = cnMatrixGet(&F, i, j);
                    FLT deriv_n = cnMatrixGet(&F_calc, i, j);
                    FLT diff_abs = fabs(deriv_n - deriv_u);
                    FLT diff_rel = diff_abs / (deriv_n + deriv_u);

                    if (diff_abs > 1e-2 && diff_rel > 1e-2) {
                        fprintf(stderr, "%2d %+7.7f %+7.7f %+7.7f %+7.7f %+7.7f \n", i, cn_as_vector(x_k_k)[i], deriv_u,
                                deriv_n, diff_abs, diff_rel);
                    }
                }
            }
            fprintf(stderr, "FJAC DEBUG END\n");
            cn_matrix_copy(&F, &F_calc);
        }

        assert(cn_is_finite(&F));

        // Run predict
        survive_kalman_predict_covariance(dt, &F, x_k_k, k);
        CN_FREE_STACK_MAT(F);
    }

    k->t = t;
}
static FLT survive_kalman_predict_update_state_extended_adaptive_internal(
	FLT t, survive_kalman_state_t *k, void *user, const struct CnMat *Z, CnMat *R, survive_kalman_meas_model_t *mk,
	struct survive_kalman_update_extended_stats_t *stats) {
	assert(R->rows == Z->rows && (R->cols == 1 || R->cols == R->rows));
	assert(Z->cols == 1);

	if (R->cols == R->rows) {
		assert(sane_covariance(R));
	}
	kalman_measurement_model_fn_t Hfn = mk->Hfn;
	bool adaptive = mk->adaptive;

    int state_cnt = k->state_cnt;

    FLT dt = t - k->t;
    assert(dt >= 0);

    FLT result = 0;

    // Setup the R matrix.
	if (adaptive && R->rows != R->cols) {
		assert(false);
		adaptive = false;
	}

	// Anything coming in this soon is liable to spike stuff since dt is so small
    if (dt < 1e-5) {
        dt = 0;
        t = k->t;
    }

    CN_CREATE_STACK_MAT(Pm, state_cnt, state_cnt);
    // Adaptive update happens on the covariance matrix prior; so save it.
	if (adaptive) {
		cn_matrix_copy(&Pm, &k->P);
	}

	CnMat *x_k_k = &k->state;
	CN_CREATE_STACK_MAT(x_k1_k1, state_cnt, 1);
	cn_matrix_copy(&x_k1_k1, x_k_k);

	// Run prediction steps -- gets new state, and covariance matrix based on time delta
	CN_CREATE_STACK_MAT(x_k_k1, state_cnt, 1);

	survive_kalman_predict(t, k, &x_k1_k1, &x_k_k1);
	if (dt > 0) {
        CN_CREATE_STACK_MAT(F, state_cnt, state_cnt);
        cn_set_constant(&F, NAN);

		k->F_fn(k->user, dt, &F, &x_k1_k1);

		if(k->debug_transition_jacobian) {
            CN_CREATE_STACK_MAT(F_calc, F.rows, F.cols);

            numeric_jacobian_predict(k, dt, &x_k1_k1, &F_calc);
            fprintf(stderr, "FJAC DEBUG BEGIN predict %d\n", x_k_k1.rows);

            for (int j = 0; j < F.cols; j++) {
                fprintf(stderr, "FJAC PARM %d\n", j);
                for (int i = 0; i < F.rows; i++) {

                    FLT deriv_u = cnMatrixGet(&F, i, j);
                    FLT deriv_n = cnMatrixGet(&F_calc, i, j);
                    FLT diff_abs = fabs(deriv_n - deriv_u);
                    FLT diff_rel = diff_abs / (deriv_n + deriv_u);

                    if (diff_abs > 1e-2 && diff_rel > 1e-2) {
                        fprintf(stderr, "%2d %+7.7f %+7.7f %+7.7f %+7.7f %+7.7f \n", i, cn_as_vector(&x_k_k1)[i], deriv_u,
                                deriv_n, diff_abs, diff_rel);
                    }
                }
            }
            fprintf(stderr, "FJAC DEBUG END\n");
            cn_matrix_copy(&F, &F_calc);
        }

		assert(cn_is_finite(&F));

        // Run predict
		survive_kalman_predict_covariance(dt, &F, &x_k_k1, k);
		CN_FREE_STACK_MAT(F);
	}

	if (k->log_level > KALMAN_LOG_LEVEL) {
		fprintf(stdout, "INFO kalman_predict_update_state_extended t=%f dt=%f ", t, dt);
		cn_print_mat(k, "Z", Z, false);
		fprintf(stdout, "\n");
	}

	CN_CREATE_STACK_MAT(K, state_cnt, Z->rows);
	CN_CREATE_STACK_MAT(HStorage, Z->rows, state_cnt);
	struct CnMat *H = &HStorage;
	if (mk->term_criteria.max_iterations > 0) {
		result = survive_kalman_run_iterations(k, Z, R, mk, user, &x_k_k1, &K, H, x_k_k, stats);
		if (result < 0)
			return result;
	} else {
		CN_CREATE_STACK_MAT(y, Z->rows, Z->cols);
		H = survive_kalman_find_residual(mk, user, Z, &x_k_k1, &y, H);

		if (H == 0) {
			return -1;
		}

		// Run update; filling in K
		survive_kalman_find_k(k, &K, H, R);

		// Calculate the next state
		cnGEMM(&K, &y, 1, &x_k_k1, 1, x_k_k, 0);
		result = cnNorm2(&y);
	}

	assert(cn_is_finite(H));
	assert(cn_is_finite(&K));

	if (k->log_level > KALMAN_LOG_LEVEL) {
		fprintf(stdout, "INFO kalman_update to    ");
		cn_print_mat(k, "x1", x_k_k, false);
	}

	survive_kalman_update_covariance(k, &K, H, R);

	if (adaptive) {
		// https://arxiv.org/pdf/1702.00884.pdf
		CN_CREATE_STACK_MAT(y, Z->rows, Z->cols);
		CN_CREATE_STACK_MAT(scaled_eTeHPkHt, Z->rows, Z->rows);
		CN_CREATE_STACK_MAT(yyt, Z->rows, Z->rows);

		survive_kalman_find_residual(mk, user, Z, x_k_k, &y, 0);
		cnMulTransposed(&y, &yyt, false, 0, 1);

		CN_CREATE_STACK_MAT(Pk_k1Ht, state_cnt, H->rows);

		FLT a = .3;
		FLT b = 1 - a;
		cnGEMM(&Pm, H, 1, 0, 0, &Pk_k1Ht, CN_GEMM_FLAG_B_T);
		cnGEMM(H, &Pk_k1Ht, b, &yyt, b, &scaled_eTeHPkHt, 0);

		cn_print_mat_v(k, 200, "PkHt", &Pk_k1Ht, true);
		cn_print_mat_v(k, 200, "yyt", &yyt, true);

		cnAddScaled(R, R, a, &scaled_eTeHPkHt, 1);

		cn_print_mat_v(k, 200, "Adaptive R", R, true);

		CN_FREE_STACK_MAT(Pk_k1Ht);
		CN_FREE_STACK_MAT(yyt);
		CN_FREE_STACK_MAT(scaled_eTeHPkHt);
	}

	k->t = t;

	CN_FREE_STACK_MAT(K);
	CN_FREE_STACK_MAT(HStorage);
	CN_FREE_STACK_MAT(x_k_k1);
	CN_FREE_STACK_MAT(Pm);

	return result;
}

FLT survive_kalman_meas_model_predict_update_stats(FLT t, struct survive_kalman_meas_model *mk, void *user,
												   const struct CnMat *Z, CnMat *R,
												   struct survive_kalman_update_extended_stats_t *stats) {
	return survive_kalman_predict_update_state_extended_adaptive_internal(t, mk->k, user, Z, R, mk, stats);
}

FLT survive_kalman_meas_model_predict_update(FLT t, struct survive_kalman_meas_model *mk, void *user,
											 const struct CnMat *Z, CnMat *R) {
	struct survive_kalman_update_extended_stats_t stats = {.total_stats = &mk->stats};
	return survive_kalman_predict_update_state_extended_adaptive_internal(t, mk->k, user, Z, R, mk, &stats);
}

/*
FLT survive_kalman_predict_update_state_extended(FLT t, survive_kalman_state_t *k, const struct CnMat *Z, CnMat* R,
												 const survive_kalman_update_extended_params_t *extended_params,
												 struct survive_kalman_update_extended_stats_t *stats) {
	return survive_kalman_predict_update_state_extended_adaptive_internal(t, k, Z, (FLT *)R, extended_params, stats);
}

FLT survive_kalman_predict_update_state(FLT t, survive_kalman_state_t *k, const struct CnMat *Z, const struct CnMat *H,
										CnMat* R, bool adaptive) {
	survive_kalman_update_extended_params_t params = {.user = (void *)H, .adapative = adaptive};
	return survive_kalman_predict_update_state_extended(t, k, Z, R, &params, 0);
}
*/
FLT survive_kalman_predict_update_state(FLT t, survive_kalman_state_t *k, const struct CnMat *Z, const struct CnMat *H,
										CnMat *R, bool adaptive) {
	survive_kalman_meas_model_t mk = {.adaptive = adaptive, .k = k};
	return survive_kalman_meas_model_predict_update(t, &mk, (void *)H, Z, R);
}

void survive_kalman_extrapolate_state(FLT t, const survive_kalman_state_t *k, size_t start_index, size_t end_index,
                                      FLT *out) {
	CN_CREATE_STACK_MAT(tmpOut, k->state_cnt, 1);
	const CnMat *x = &k->state;

	FLT dt = t == 0. ? 0 : t - k->t;
	const FLT *copyFrom = cn_as_const_vector(&k->state);
	if (dt > 0) {
		k->Predict_fn(dt, k, x, &tmpOut);
		copyFrom = _tmpOut;
	}
	assert(out != copyFrom);
	memcpy(out, copyFrom + start_index, (end_index - start_index) * sizeof(FLT));
	CN_FREE_STACK_MAT(tmpOut);
}
void survive_kalman_set_P(survive_kalman_state_t *k, const FLT *p) { cn_set_diag(&k->P, p); }
