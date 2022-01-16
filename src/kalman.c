#include "cnkalman/kalman.h"
#if !defined(__FreeBSD__) && !defined(__APPLE__)
#include <malloc.h>
#endif
#include <memory.h>
#include <cnkalman/numerical_diff.h>
#include <cnmatrix/cn_matrix.h>
#include <stdio.h>
#include <cnkalman/kalman.h>


#include "cnkalman_internal.h"

#include "math.h"

#define KALMAN_LOG_LEVEL 1000

#define CN_KALMAN_VERBOSE(lvl, fmt, ...)                                                                               \
	{                                                                                                                  \
		if (k->log_level >= lvl) {                                                                                     \
			fprintf(stdout, fmt "\n", __VA_ARGS__);                                                                    \
		}                                                                                                              \
	}

void cnkalman_set_logging_level(cnkalman_state_t *k, int v) { k->log_level = v; }

void cnkalman_linear_update(struct CnMat *F, const struct CnMat *x0, struct CnMat *x1) {
	cnGEMM(F, x0, 1, 0, 0, x1, 0);
}
void kalman_print_mat_v(const cnkalman_state_t *k, int ll, const char *name, const CnMat *M, bool newlines) {
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
				fprintf(stdout, "             0, ");
			else
				fprintf(stdout, "%+7.7e,", v);
		}
		if (newlines && M->cols > 1)
			fprintf(stdout, "\n");
	}
	fprintf(stdout, "\n");
}
static void kalman_print_mat(cnkalman_state_t *k, const char *name, const CnMat *M, bool newlines) {
	kalman_print_mat_v(k, KALMAN_LOG_LEVEL, name, M, newlines);
}

void user_is_q(void *user, FLT t, const struct CnMat *x, CnMat *Q_out) {
	const CnMat *q = (const CnMat *)user;
	cnScale(Q_out, q, t);
}

CN_EXPORT_FUNCTION void cnkalman_state_reset(cnkalman_state_t *k) {
	k->t = 0;
	cn_set_zero(&k->P);

	cnkalman_predict_state(10, k);
	k->t = 0;

	kalman_print_mat(k, "initial Pk_k", &k->P, true);
}

CN_EXPORT_FUNCTION void cnkalman_meas_model_init(cnkalman_state_t *k, const char *name,
												   cnkalman_meas_model_t *mk, kalman_measurement_model_fn_t Hfn) {
	memset(mk, 0, sizeof(*mk));
	mk->k = k;
	mk->name = name;
	mk->Hfn = Hfn;
	mk->term_criteria = (struct term_criteria_t){.max_iterations = 0, .xtol = 1e-2, .mtol = 1e-8, .minimum_step = .05};
}

void cnkalman_error_state_init(cnkalman_state_t *k, size_t state_cnt, size_t error_state_cnt,
							   kalman_transition_model_fn_t F, kalman_process_noise_fn_t q_fn,
							   kalman_error_state_model_fn_t Err_F, void *user, double *state) {
	memset(k, 0, sizeof(*k));

	k->state_cnt = (int)state_cnt;
	k->error_state_size = error_state_cnt;
	k->ErrorState_fn = Err_F;

	k->Q_fn = q_fn;

	size_t p_size = Err_F ? error_state_cnt : state_cnt;
	k->P = cnMatCalloc(p_size, p_size);

	k->Transition_fn = F;
	k->user = user;

	if (!state) {
		k->State_is_heap = true;
		state = (FLT*)calloc(1, sizeof(FLT) * k->state_cnt);
	}

	k->state = cnMat(k->state_cnt, 1, state);
}

void cnkalman_state_init(cnkalman_state_t *k, size_t state_cnt, kalman_transition_model_fn_t F,
							   kalman_process_noise_fn_t q_fn, void *user, FLT *state) {
	cnkalman_error_state_init(k, state_cnt, state_cnt, F, q_fn, 0, user, state);
}

void cnkalman_state_free(cnkalman_state_t *k) {
	free(k->P.data);
	k->P.data = 0;

	if (k->State_is_heap)
		free(CN_FLT_PTR(&k->state));
	k->state.data = 0;
}

void cnkalman_predict_covariance(FLT dt, const CnMat *F, const CnMat *x, cnkalman_state_t *k) {
	int dims = k->state_cnt;

	CnMat *Pk1_k1 = &k->P;
	kalman_print_mat(k, "Pk-1_k-1", Pk1_k1, 1);

	// k->P = F * k->P * F^T + Q
	if(k->state_variance_per_second.rows > 0) {
		cn_add_diag(Pk1_k1, &k->state_variance_per_second, dt);
		assert(sane_covariance(Pk1_k1));
	}

	struct CnMat* Qp = 0;
	CN_CREATE_STACK_MAT(Q, dims, dims);
	if(k->Q_fn) {
		k->Q_fn(k->user, dt, x, &Q);
		Qp = &Q;
		assert(sane_covariance(&Q));
	}
	cn_ABAt_add(Pk1_k1, F, Pk1_k1, Qp);
	assert(sane_covariance(Pk1_k1));

	// printf("!!!! %f\n", cnDet(Pk1_k1));
	// assert(cnDet(Pk1_k1) >= 0);
	if (k->log_level >= KALMAN_LOG_LEVEL) {
		CN_KALMAN_VERBOSE(110, "T: %f", dt);
		kalman_print_mat(k, "Q", Qp, 1);
		kalman_print_mat(k, "F", F, 1);
		kalman_print_mat(k, "Pk1_k-1", Pk1_k1, 1);
	}
	CN_FREE_STACK_MAT(Q);
}
void cnkalman_find_k(cnkalman_state_t *k, cnkalman_gain_matrix *K, const struct CnMat *H,
								  const CnMat *R) {
	int dims = k->state_cnt;

	const CnMat *Pk_k = &k->P;

	CN_CREATE_STACK_MAT(Pk_k1Ht, Pk_k->rows, H->rows);

	// Pk_k1Ht = P_k|k-1 * H^T
	cnGEMM(Pk_k, H, 1, 0, 0, &Pk_k1Ht, CN_GEMM_FLAG_B_T);
	CN_CREATE_STACK_MAT(S, H->rows, H->rows);

	kalman_print_mat(k, "H", H, 1);
	kalman_print_mat(k, "R", R, 1);

	// S = H * P_k|k-1 * H^T + R
	if (R->cols == 1) {
		cnGEMM(H, &Pk_k1Ht, 1, 0, 0, &S, 0);
		for (int i = 0; i < S.rows; i++)
			cnMatrixSet(&S, i, i, cnMatrixGet(&S, i, i) + cn_as_const_vector(R)[i]);
	} else {
		cnGEMM(H, &Pk_k1Ht, 1, R, 1, &S, 0);
	}

	assert(cn_is_finite(&S));

	kalman_print_mat(k, "Pk_k1Ht", &Pk_k1Ht, 1);
	kalman_print_mat(k, "S", &S, 1);

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
	kalman_print_mat(k, "iS", &iS, 1);

	// K = Pk_k1Ht * iS
	cnGEMM(&Pk_k1Ht, &iS, 1, 0, 0, K, 0);
	kalman_print_mat(k, "K", K, 1);
}

static void cnkalman_update_covariance(cnkalman_state_t *k, const cnkalman_gain_matrix *K,
											 const struct CnMat *H, const struct CnMat *R) {
	int filter_cnt = k->P.rows;
	CN_CREATE_STACK_MAT(eye, filter_cnt, filter_cnt);
	cn_set_diag_val(&eye, 1);

	CN_CREATE_STACK_MAT(ikh, filter_cnt, filter_cnt);

	// ikh = (I - K * H)
	cnGEMM(K, H, -1, &eye, 1, &ikh, 0);

	// cvGEMM does not like the same addresses for src and destination...
	CnMat *Pk_k = &k->P;
	CN_CREATE_STACK_MAT(tmp, filter_cnt, filter_cnt);
	cnCopy(Pk_k, &tmp, 0);

	CN_CREATE_STACK_MAT(kRkt, filter_cnt, filter_cnt);
	bool use_joseph_form = R->rows == R->cols;
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
		kalman_print_mat(k, "K", K, true);

		kalman_print_mat(k, "ikh", &ikh, true);

		fprintf(stdout, "INFO new Pk_k\t");
		kalman_print_mat(k, "Pk_k", Pk_k, true);
	}
	CN_FREE_STACK_MAT(tmp);
	CN_FREE_STACK_MAT(ikh);
	CN_FREE_STACK_MAT(eye);
}

static inline void cnkalman_predict(FLT t, cnkalman_state_t *k, const CnMat *x_t0_t0, CnMat *x_t0_t1, CnMat* F) {
	// X_k|k-1 = Predict(X_K-1|k-1)
	if (k->log_level > KALMAN_LOG_LEVEL) {
		fprintf(stdout, "INFO kalman_predict from ");
		kalman_print_mat(k, "x_t0_t0", x_t0_t0, false);
	}
	assert(cn_as_const_vector(x_t0_t0) != cn_as_const_vector(x_t0_t1));
	if (t == k->t) {
		cnCopy(x_t0_t0, x_t0_t1, 0);
	} else {
		assert(t > k->t);
		k->Transition_fn(t - k->t, k, x_t0_t0, x_t0_t1, F);
	}
	if (k->log_level > KALMAN_LOG_LEVEL) {
		fprintf(stdout, "INFO kalman_predict to   ");
		kalman_print_mat(k, "x_t0_t1", x_t0_t1, false);
	}
	if (k->datalog) {
		CN_CREATE_STACK_MAT(tmp, x_t0_t0->rows, x_t0_t1->cols);
		cn_elementwise_subtract(&tmp, x_t0_t1, x_t0_t0);
		k->datalog(k, "predict_diff", cn_as_const_vector(&tmp), tmp.rows * tmp.cols);
	}
}

typedef struct numeric_jacobian_predict_fn_ctx {
    FLT dt;
    cnkalman_state_t *k;
} numeric_jacobian_predict_fn_ctx;
static bool numeric_jacobian_predict_fn(void * user, const struct CnMat *x, struct CnMat *y) {
    numeric_jacobian_predict_fn_ctx* ctx = user;
    ctx->k->Transition_fn(ctx->dt, ctx->k, x, y, 0);
    return true;
}

static bool numeric_jacobian_predict(cnkalman_state_t *k, enum cnkalman_jacobian_mode mode, FLT dt, const struct CnMat *x, CnMat *H) {
    numeric_jacobian_predict_fn_ctx ctx = {
            .dt = dt,
            .k = k
    };
    return cnkalman_numerical_differentiate(&ctx, mode == cnkalman_jacobian_mode_debug ?
    (enum cnkalman_numerical_differentiate_mode) cnkalman_jacobian_mode_two_sided : mode, numeric_jacobian_predict_fn, x, H);
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

static bool numeric_jacobian(enum cnkalman_jacobian_mode mode, kalman_measurement_model_fn_t Hfn, void *user, const struct CnMat *Z, const struct CnMat *x, CnMat *H) {
    numeric_jacobian_meas_fn_ctx ctx = {
            .Hfn = Hfn,
            .user = user,
            .Z = Z
    };
    return cnkalman_numerical_differentiate(&ctx,mode == cnkalman_jacobian_mode_debug ?
    cnkalman_numerical_differentiate_mode_two_sided : mode, numeric_jacobian_meas_fn, x, H);
}

static inline bool compare_jacobs(const char* label, const CnMat *H, const CnMat *H_calc, const CnMat *y, const CnMat *Z) {
	bool needsPrint = true;
	fprintf(stderr, "FJAC DEBUG BEGIN %s %2dx%2d\n", label, H->rows, H->cols);

    for (int j = 0; j < H->cols; j++) {
		if(!needsPrint) {
			fprintf(stderr, "FJAC COLUMN %d\n", j);
		}
        for (int i = 0; i < H->rows; i++) {

            FLT deriv_u = cnMatrixGet(H, i, j);
            FLT deriv_n = cnMatrixGet(H_calc, i, j);
            FLT diff_abs = fabs(deriv_n - deriv_u);
            FLT diff_rel = diff_abs / (deriv_n + deriv_u + 1e-10);

            if (needsPrint == false || (diff_abs > 1e-2 && diff_rel > 1e-2)) {
				if(needsPrint) {
					fprintf(stderr, "FJAC DEBUG BEGIN %s %2dx%2d\n", label, H->rows, H->cols);
					fprintf(stderr, "FJAC COLUMN %d\n", j);

					needsPrint = false;
				}

                fprintf(stderr, "%2d %+7.7f %+7.7f %+7.7f %+7.7f %+7.7f %+7.7f \n", i, cn_as_const_vector(Z)[i],
						cn_as_const_vector(y)[i], deriv_u,
                        deriv_n, diff_abs, diff_rel);
            }
        }
    }
	if(!needsPrint) {
		fprintf(stderr, "FJAC DEBUG END\n");
	}
	return needsPrint;
}

CnMat *cnkalman_find_residual(cnkalman_meas_model_t *mk, void *user, const struct CnMat *Z,
										   const struct CnMat *x, CnMat *y, CnMat *H) {
	cnkalman_state_t *k = mk->k;
	kalman_measurement_model_fn_t Hfn = mk->Hfn;

	if (H) {
		cn_set_constant(H, INFINITY);
	}

	CnMat *rtn = 0;
	CN_CREATE_STACK_MAT(HFullState, Z->rows, k->ErrorState_fn ? k->state_cnt : 0);

	if (Hfn) {
        bool okay = Hfn(user, Z, x, y, k->ErrorState_fn ? &HFullState : H);
		if (okay == false) {
			return 0;
		}

		if (mk->meas_jacobian_mode != cnkalman_jacobian_mode_user_fn && H) {
			CN_CREATE_STACK_MAT(H_calc, H->rows, H->cols);

			numeric_jacobian(mk->meas_jacobian_mode, Hfn, user, Z, x, &H_calc);

			if(mk->meas_jacobian_mode == cnkalman_jacobian_mode_debug) {
                if(!compare_jacobs(mk->name, H, &H_calc, y, Z)) {
					fprintf(stderr, "For state: \n");
					cn_print_mat(x);
				}
            }

            cn_matrix_copy(H, &H_calc);
		}
		rtn = H;
	} else {
		rtn = (struct CnMat *)user;
		cnGEMM(rtn, x, -1, Z, 1, y, 0);
	}

	if(k->Update_fn && rtn && H) {
		CN_CREATE_STACK_MAT(Hxsx, k->state_cnt, k->error_state_size);
		k->Update_fn(user, x, 0, 0, &Hxsx);
		cnGEMM(Hfn ? &HFullState : rtn, &Hxsx, 1, 0, 0, H, 0);
		rtn = H;
	}

	assert(!rtn || cn_is_finite(rtn));
	assert(cn_is_finite(y));

	return rtn;
}

void cnkalman_predict_state(FLT t, cnkalman_state_t *k) {
    FLT dt = t - k->t;
    assert(dt >= 0);

    int state_cnt = k->state_cnt;
	int filter_cnt = k->error_state_size;
    CnMat *x_k_k = &k->state;

    CN_CREATE_STACK_MAT(x_k1_k1, state_cnt, 1);
    cn_matrix_copy(&x_k1_k1, x_k_k);


    if (dt > 0) {
		size_t f_size = k->error_state_transition ? k->error_state_size : state_cnt;
        CN_CREATE_STACK_MAT(F, f_size, f_size);
		CN_CREATE_STACK_MAT(FHxsx, k->error_state_size, k->error_state_size);
		cn_set_constant(&F, NAN);

        cnkalman_predict(t, k, &x_k1_k1, x_k_k, &F);

		assert(cn_is_finite(&F));

		CnMat* FP = &F;
		if(k->Update_fn && k->error_state_transition == false) {
			CN_CREATE_STACK_MAT(OldXOlde, state_cnt, k->error_state_size);
			k->Update_fn(k->user, &x_k1_k1, 0, 0, &OldXOlde);
			CN_CREATE_STACK_MAT(NewENewX, k->error_state_size, state_cnt);
			k->ErrorState_fn(k->user, x_k_k, 0, 0, &NewENewX);

			CN_CREATE_STACK_MAT(FOldXOldE, F.rows, OldXOlde.cols);
			cnGEMM(&F, &OldXOlde, 1, 0, 0, &FOldXOldE, 0);
			cnGEMM(&NewENewX, &FOldXOldE, 1, 0, 0, &FHxsx, 0);
			FP = &FHxsx;
		}

		if(k->transition_jacobian_mode != cnkalman_jacobian_mode_user_fn) {
            CN_CREATE_STACK_MAT(F_calc, F.rows, F.cols);

            numeric_jacobian_predict(k, k->transition_jacobian_mode, dt, &x_k1_k1, &F_calc);

            if(k->transition_jacobian_mode == cnkalman_jacobian_mode_debug) {
                compare_jacobs("predict", &F, &F_calc, &x_k1_k1, x_k_k);
            }

            cn_matrix_copy(&F, &F_calc);
        }

        // Run predict
		cnkalman_predict_covariance(dt, FP, x_k_k, k);
        CN_FREE_STACK_MAT(F);
    }

    k->t = t;
}

// https://arxiv.org/pdf/1702.00884.pdf
static void
calculate_adaptive_covariance(cnkalman_meas_model_t *mk, void *user, const struct CnMat *Z, CnMat *R,
                              CnMat *Pm, const struct CnMat *H) {
    const cnkalman_state_t *k = mk->k;
    const CnMat *x_k_k = &k->state;
    int state_cnt = k->state_cnt;

    CN_CREATE_STACK_MAT(y, Z->rows, Z->cols);
    CN_CREATE_STACK_MAT(scaled_eTeHPkHt, Z->rows, Z->rows);
    CN_CREATE_STACK_MAT(yyt, Z->rows, Z->rows);

    cnkalman_find_residual(mk, user, Z, x_k_k, &y, 0);
    cnMulTransposed(&y, &yyt, false, 0, 1);

    CN_CREATE_STACK_MAT(Pk_k1Ht, state_cnt, H->rows);

    FLT a = .3;
    FLT b = 1 - a;
    cnGEMM(Pm, H, 1, 0, 0, &Pk_k1Ht, CN_GEMM_FLAG_B_T);
    cnGEMM(H, &Pk_k1Ht, b, &yyt, b, &scaled_eTeHPkHt, 0);

    kalman_print_mat_v(k, 200, "PkHt", &Pk_k1Ht, true);
    kalman_print_mat_v(k, 200, "yyt", &yyt, true);

    cnAddScaled(R, R, a, &scaled_eTeHPkHt, 1);

    kalman_print_mat_v(k, 200, "Adaptive R", R, true);

    CN_FREE_STACK_MAT(Pk_k1Ht);CN_FREE_STACK_MAT(yyt);CN_FREE_STACK_MAT(scaled_eTeHPkHt);
}

static FLT cnkalman_predict_update_state_extended_adaptive_internal(
	FLT t, cnkalman_state_t *k, void *user, const struct CnMat *Z, CnMat *R, cnkalman_meas_model_t *mk,
	struct cnkalman_update_extended_stats_t *stats) {
	assert(R->rows == Z->rows && (R->cols == 1 || R->cols == R->rows));
	assert(Z->cols == 1);

	if (R->cols == R->rows) {
		assert(sane_covariance(R));
	}
	kalman_measurement_model_fn_t Hfn = mk->Hfn;
	bool adaptive = mk->adaptive;

    int state_cnt = k->state_cnt;
	int filter_cnt = k->error_state_size;

    FLT dt = t - k->t;

	// Allow very small negative integrations
    assert(dt >= -.01);
	if(dt < 0) {
		dt = 0;
		t = k->t;
	}
    FLT result = 0;

    // Setup the R matrix.
	if (adaptive && R->rows != R->cols) {
		assert(false);
		adaptive = false;
	}

	CnMat *x_k_k = &k->state;

	// Run prediction steps -- gets new state, and covariance matrix based on time delta
	CN_CREATE_STACK_VEC(x_k_k1, state_cnt);
    cnkalman_predict_state(t, k);
    cn_matrix_copy(&x_k_k1, &k->state);

    // Adaptive update happens on the covariance matrix prior; so save it.
    CN_CREATE_STACK_MAT(Pm, state_cnt, state_cnt);
    if (adaptive) {
        cn_matrix_copy(&Pm, &k->P);
    }

	if (k->log_level > KALMAN_LOG_LEVEL) {
		fprintf(stdout, "INFO kalman_predict_update_state_extended t=%f dt=%f ", t, dt);
		kalman_print_mat(k, "Z", Z, false);
		fprintf(stdout, "\n");
	}

	CN_CREATE_STACK_MAT(K, filter_cnt, Z->rows);
	CN_CREATE_STACK_MAT(HStorage, Z->rows, filter_cnt);
	struct CnMat *H = &HStorage;

	if (mk->term_criteria.max_iterations > 0) {
		result = cnkalman_run_iterations(k, Z, R, mk, user, &x_k_k1, &K, H, x_k_k, stats);
		if (result < 0)
			return result;
	} else {
        CN_CREATE_STACK_MAT(y, Z->rows, Z->cols);
		H = cnkalman_find_residual(mk, user, Z, &x_k_k1, &y, H);

		if (H == 0) {
			return -1;
		}

		// Run update; filling in K
		cnkalman_find_k(k, &K, H, R);

		// Calculate the next state
		CN_CREATE_STACK_VEC(Ky, filter_cnt);
		cnGEMM(&K, &y, 1, 0, 0, &Ky, 0);
		cnkalman_update_state(user, k, &x_k_k1, 1, &Ky, x_k_k);
		result = cnNorm2(&y);

		if(stats) {
            CN_CREATE_STACK_MAT(yp, Z->rows, Z->cols);
            cnkalman_find_residual(mk, user, Z, x_k_k, &yp, 0);
		    stats->origerror = sqrt(result);
            stats->besterror = cnNorm(&yp);
		}
	}

	assert(cn_is_finite(H));
	assert(cn_is_finite(&K));

	if (k->log_level > KALMAN_LOG_LEVEL) {
		fprintf(stdout, "INFO kalman_update to    ");
		kalman_print_mat(k, "x1", x_k_k, false);
	}

	cnkalman_update_covariance(k, &K, H, R);
	assert(sane_covariance(&k->P));

	if (adaptive) {
        calculate_adaptive_covariance(mk, user, Z, R, &Pm, H);
    }

	k->t = t;

	CN_FREE_STACK_MAT(K);
	CN_FREE_STACK_MAT(HStorage);
	CN_FREE_STACK_MAT(x_k_k1);
	CN_FREE_STACK_MAT(Pm);

	return result;
}

FLT cnkalman_meas_model_predict_update_stats(FLT t, struct cnkalman_meas_model *mk, void *user,
												   const struct CnMat *Z, CnMat *R,
												   struct cnkalman_update_extended_stats_t *stats) {
	return cnkalman_predict_update_state_extended_adaptive_internal(t, mk->k, user, Z, R, mk, stats);
}

FLT cnkalman_meas_model_predict_update(FLT t, struct cnkalman_meas_model *mk, void *user,
											 const struct CnMat *Z, CnMat *R) {
	struct cnkalman_update_extended_stats_t stats = {.total_stats = &mk->stats};
	return cnkalman_predict_update_state_extended_adaptive_internal(t, mk->k, user, Z, R, mk, &stats);
}

FLT cnkalman_predict_update_state(FLT t, cnkalman_state_t *k, const struct CnMat *Z, const struct CnMat *H,
										CnMat *R, bool adaptive) {
	cnkalman_meas_model_t mk = {.adaptive = adaptive, .k = k};
	return cnkalman_meas_model_predict_update(t, &mk, (void *)H, Z, R);
}

void cnkalman_extrapolate_state(FLT t, const cnkalman_state_t *k, size_t start_index, size_t end_index,
                                      FLT *out) {
	CN_CREATE_STACK_MAT(tmpOut, k->state_cnt, 1);
	const CnMat *x = &k->state;

	FLT dt = t == 0. ? 0 : t - k->t;
	const FLT *copyFrom = cn_as_const_vector(&k->state);
	if (dt > 0) {
		k->Transition_fn(dt, k, x, &tmpOut, 0);
		copyFrom = _tmpOut;
	}
	assert(out != copyFrom);
	memcpy(out, copyFrom + start_index, (end_index - start_index) * sizeof(FLT));
	CN_FREE_STACK_MAT(tmpOut);
}
void cnkalman_set_P(cnkalman_state_t *k, const FLT *p) { cn_set_diag(&k->P, p); }
