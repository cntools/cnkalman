#pragma once

#include <string>
#include <cnmatrix/cn_matrix.h>
#include <cnkalman/survive_kalman.h>

struct Model {
    std::string id;
    size_t state_cnt, meas_cnt;

    virtual void transition(FLT dt, CnMat& cF, const CnMat& x) = 0;
    virtual void measurement_mdl(const CnMat& x, CnMat* z, CnMat* h) = 0;
    virtual void predict(FLT dt, const struct CnMat &x0, struct CnMat &x1) {
        int state_cnt = this->state_cnt;
        CN_CREATE_STACK_MAT(F, state_cnt, state_cnt);
        transition(dt, F, x0);

        // X_k|k-1 = F * X_k-1|k-1
        cnGEMM(&F, &x0, 1, 0, 0, &x1, (cnGEMMFlags)0);
        CN_FREE_STACK_MAT(F);
    }
    virtual void process_noise(FLT dt, const struct CnMat *x, struct CnMat *Q_out) = 0;
    virtual void sample_measurement(FLT dt, struct CnMat &Z, struct CnMat &R) {
        CN_CREATE_STACK_MAT(RL, meas_cnt, meas_cnt);
        cnSqRoot(&R, &RL);
        CN_CREATE_STACK_MAT(X, meas_cnt, 1);
        CN_CREATE_STACK_MAT(Xs, meas_cnt, 1);
        cnRand(&X, 0, 1);
        cnGEMM(&RL, &X, 1, 0, 0, &Xs, (cnGEMMFlags)0);
        //cn_elementwise_add(&Z, &Z, &Xs);
    }
    virtual void sample_state(FLT dt, const struct CnMat &x0, struct CnMat &x1) {

    }
};

template <typename M>
struct ModelRunner {
    survive_kalman_state_t kalman_state = {};
    survive_kalman_meas_model meas_mdl = {};
    M model;
    FLT* state;
    CnMat stateM;

    static void kalman_predict_bounce(FLT dt, const struct survive_kalman_state_s *k, const struct CnMat *x0,
                                        struct CnMat *x1) {
        ModelRunner<M>* self = (ModelRunner<M>*)k->user;
        self->model.predict(dt, *x0, *x1);
    }

    static void transition_bounce(void* user, FLT dt, struct CnMat *f_out, const struct CnMat *x0) {
        ModelRunner<M>* self = (ModelRunner<M>*)user;
        self->model.transition(dt, *f_out, *x0);
    }
    static void kalman_process_noise_fn_t(void *user, FLT dt, const struct CnMat *x, struct CnMat *Q_out) {
        ModelRunner<M>* self = (ModelRunner<M>*)user;
        self->model.process_noise(dt, x, Q_out);
    }

    static bool kalman_measurement(void *user, const struct CnMat *Z, const struct CnMat *x_t, struct CnMat *y, struct CnMat *H_k) {
        ModelRunner<M>* self = (ModelRunner<M>*)user;

        CN_CREATE_STACK_VEC(pz, self->model.meas_cnt);
        self->model.measurement_mdl(*x_t, &pz, H_k);
        cn_elementwise_subtract(y, Z, &pz);

        return true;
    }

    ModelRunner() {
        survive_kalman_state_init(&kalman_state, model.state_cnt, transition_bounce, kalman_process_noise_fn_t, this, 0);
        kalman_state.Predict_fn = kalman_predict_bounce;
        survive_kalman_meas_model_init(&kalman_state, "meas", &meas_mdl, kalman_measurement);
        state = kalman_state.state.data;
        stateM = cnVec(model.state_cnt, state);
    }

    survive_kalman_update_extended_stats_t predict_update(FLT t, const struct CnMat& Z, CnMat& R) {
        struct survive_kalman_update_extended_stats_t stats = {};
        survive_kalman_meas_model_predict_update_stats(t, &meas_mdl, this, &Z, &R, &stats);
        return stats;
    }
    void predict(FLT t) {
        survive_kalman_predict_state(t, &kalman_state);
    }
};