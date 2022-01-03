#include <cnkalman/kalman.h>
#include <stdio.h>
#include "cnkalman_internal.h"

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

    return V.data[0];
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

const char *cnkalman_update_extended_termination_reason_to_str(
        enum cnkalman_update_extended_termination_reason reason) {
    switch (reason) {
        case cnkalman_update_extended_termination_reason_none:
            return "none";
        case cnkalman_update_extended_termination_reason_maxiter:
            return "maxiter";
        case cnkalman_update_extended_termination_reason_invalid_jacobian:
            return "invalid_jac";
        case cnkalman_update_extended_termination_reason_xtol:
            return "xtol";
        case cnkalman_update_extended_termination_reason_MAX:
            return "MAX";
        case cnkalman_update_extended_termination_reason_step:
            return "step";
        case cnkalman_update_extended_termination_reason_mtol:
            return "mtol";
        default:
            return "";
    }
}

static inline enum cnkalman_update_extended_termination_reason
cnkalman_termination_criteria(cnkalman_state_t *k, const struct term_criteria_t *term_criteria,
                                    FLT initial_error, FLT error, FLT alpha, FLT last_error) {
    FLT minimum_step = term_criteria->minimum_step > 0 ? term_criteria->minimum_step : .01;
    if (alpha == 0 || alpha < minimum_step) {
        return cnkalman_update_extended_termination_reason_step;
    }
    if (error == 0) {
        return cnkalman_update_extended_termination_reason_xtol;
    }

    if (isfinite(last_error) && fabs(last_error - error) < term_criteria->xtol * error) {
        return cnkalman_update_extended_termination_reason_xtol;
    }
    return cnkalman_update_extended_termination_reason_none;
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
FLT cnkalman_run_iterations(cnkalman_state_t *k, const struct CnMat *Z, const struct CnMat *R,
                                         cnkalman_meas_model_t *mk, void *user, const CnMat *x_k_k1, CnMat *K,
                                         CnMat *H, CnMat *x_k_k, struct cnkalman_update_extended_stats_t *stats) {
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

    enum cnkalman_update_extended_termination_reason stop_reason =
            cnkalman_update_extended_termination_reason_none;
    FLT current_norm = INFINITY, last_norm = INFINITY;
    FLT initial_norm = 0;

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

    for (iter = 0; iter < max_iter && stop_reason == cnkalman_update_extended_termination_reason_none; iter++) {
        // Find the residual y and possibly also the jacobian H. The user could have passed one in as 'user', or given
        // us a map function which will calculate it.
        struct CnMat *HR = cnkalman_find_residual(mk, user, Z, &x_i, &y, H);
        if (stats) {
            stats->fevals++;
            stats->hevals++;
        }

        // If the measurement jacobian isn't calculable, the best we can do is just bail.
        if (HR == 0) {
            stop_reason = cnkalman_update_extended_termination_reason_invalid_jacobian;
            current_norm = INFINITY;
            break;
        }
        last_norm = current_norm;

        cnSub(&x_diff, x_k_k1, &x_i);
        current_norm = calculate_v(&y, &x_diff, &iR, &iP, &meas_part, iter == 0 ? 0 : &delta_part);
        assert(current_norm >= 0);

        cnGEMM(&iP, &x_diff, 1, 0, 0, &iPdx, 0);

        if (R->cols > 1) {
            cnGEMM(&iR, &y, 1, 0, 0, &iRy, 0);
        } else {
            cnElementwiseMultiply(&iRy, &iR, &y);
        }
        cnGEMM(H, &iRy, -1, &iPdx, -1, &dVt, CN_GEMM_FLAG_A_T);

        if (iter == 0) {
            initial_norm = current_norm;
            if (stats) {
                stats->orignorm_meas += meas_part;
                stats->origerror = cnNorm(&y);
            }
        }

        // Run update; filling in K
        cnkalman_find_k(k, K, H, R);

        if ((stop_reason =
                     cnkalman_termination_criteria(k, &mk->term_criteria, initial_norm, current_norm, 1, last_norm)) >
            cnkalman_update_extended_termination_reason_none) {
            goto end_of_loop;
        }

        // x_update = K * (z - y - H * Δx)
        cnGEMM(H, &x_diff, 1, 0, 0, &Hxdiff, 0);
        cnSub(&y, &y, &Hxdiff);
        cnGEMM(K, &y, 1, &x_diff, 1, &x_update, 0);

        FLT scale = 1.;
        FLT m = cnDot(&dVt, &x_update);
        if (fabs(m) < mk->term_criteria.mtol) {
            stop_reason = cnkalman_update_extended_termination_reason_mtol;
            break;
        }
        FLT c = .5, tau = .5;
        FLT fa = 0, fa_best = current_norm, a_best = 0;

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

            if (fa >= current_norm + scale * m * c) {
                exit_condition = false;
                if (fa_best > fa) {
                    fa_best = fa;
                    a_best = scale;
                }
                scale = tau * scale;

                if (scale <= min_step) {
                    current_norm = fa_best;
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
            fprintf(stdout, "%3d: %7.7f / %7.7f (%f, %f, %f) ", iter, initial_norm, current_norm, scale, m,
                    cnNorm(&x_update));
            cn_print_mat_v(k, 1000, "new x", &x_i, false);
        }
        if (stop_reason == cnkalman_update_extended_termination_reason_none)
            stop_reason =
                    cnkalman_termination_criteria(k, &mk->term_criteria, initial_norm, current_norm, scale, last_norm);
    }
    if (stop_reason == cnkalman_update_extended_termination_reason_none)
        stop_reason = cnkalman_update_extended_termination_reason_maxiter;
    bool isFailure = current_norm > initial_norm || isinf(current_norm);
    if (stats) {
        stats->besterror = cnNorm(&y);
        stats->iterations = iter;
        stats->orignorm = initial_norm;
        stats->bestnorm = current_norm;
        stats->stop_reason = stop_reason;
        stats->bestnorm_meas = meas_part;
        stats->bestnorm_delta = delta_part;
        if (stats->total_stats) {
            stats->total_stats->total_runs++;
            stats->total_stats->total_failures += isFailure;
            stats->total_stats->orignorm_acc += initial_norm;
            stats->total_stats->bestnorm_acc += current_norm;
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
        initial_norm = -1;
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

    return initial_norm;
}
