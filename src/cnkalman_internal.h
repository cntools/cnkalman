#include <cnkalman/survive_kalman.h>

typedef struct CnMat survive_kalman_gain_matrix;

CnMat *survive_kalman_find_residual(survive_kalman_meas_model_t *mk, void *user, const struct CnMat *Z,
                                    const struct CnMat *x, CnMat *y, CnMat *H);

FLT survive_kalman_run_iterations(survive_kalman_state_t *k, const struct CnMat *Z, const struct CnMat *R,
                                  survive_kalman_meas_model_t *mk, void *user, const CnMat *x_k_k1, CnMat *K,
                                  CnMat *H, CnMat *x_k_k, struct survive_kalman_update_extended_stats_t *stats);

void survive_kalman_find_k(survive_kalman_state_t *k, survive_kalman_gain_matrix *K, const struct CnMat *H,
                                  const CnMat *R);

void cn_print_mat_v(const survive_kalman_state_t *k, int ll, const char *name, const CnMat *M, bool newlines);

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
