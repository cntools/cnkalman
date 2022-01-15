#include <cnkalman/kalman.h>

typedef struct CnMat cnkalman_gain_matrix;

CnMat *cnkalman_find_residual(cnkalman_meas_model_t *mk, void *user, const struct CnMat *Z,
                                    const struct CnMat *x, CnMat *y, CnMat *H);

FLT cnkalman_run_iterations(cnkalman_state_t *k, const struct CnMat *Z, const struct CnMat *R,
                                  cnkalman_meas_model_t *mk, void *user, const CnMat *x_k_k1, CnMat *K,
                                  CnMat *H, CnMat *x_k_k, struct cnkalman_update_extended_stats_t *stats);
void cnkalman_update_state(void* user, cnkalman_state_t *k, const CnMat* x0, FLT scale, const CnMat* error_state, CnMat* x1);
void cnkalman_find_k(cnkalman_state_t *k, cnkalman_gain_matrix *K, const struct CnMat *H,
                                  const CnMat *R);

void kalman_print_mat_v(const cnkalman_state_t *k, int ll, const char *name, const CnMat *M, bool newlines);

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
