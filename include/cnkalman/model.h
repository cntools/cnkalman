#pragma once

#include "cnkalman/kalman.h"
#ifdef __cplusplus
#include <vector>
#include <memory>
#include <string>

namespace cnkalman {
    struct KalmanModel;
    struct KalmanMeasurementModel {
        size_t meas_cnt;
        cnkalman_meas_model meas_mdl = {};

        KalmanMeasurementModel(KalmanModel* kalmanModel, const std::string& name, size_t meas_cnt);
        virtual ~KalmanMeasurementModel() = default;
        virtual bool predict_measurement(const CnMat& x, CnMat* z, CnMat* h) = 0;
        virtual bool residual(const CnMat& Z, const CnMat& x, CnMat* y, CnMat* h);

        cnkalman_update_extended_stats_t update(FLT t, const struct CnMat& Z, CnMat& R);

        virtual std::ostream& write(std::ostream&) const;
        virtual void sample_measurement(const CnMat& x, struct CnMat& Z, const CnMat& R);
        virtual CnMat default_R() {
            auto rtn = cnMatCalloc(meas_cnt, meas_cnt);
            cn_set_diag_val(&rtn, .1 * .1);
            return rtn;
        }
    };

    struct KalmanModel {
        std::string name;

        cnkalman_state_t kalman_state = {};
        size_t state_cnt;
        FLT* state;
        CnMat* stateM;
        std::vector<std::shared_ptr<struct KalmanMeasurementModel>> measurementModels;

        virtual std::ostream& write(std::ostream&) const;

        KalmanModel(const std::string& name, size_t state_cnt);
        virtual ~KalmanModel();

        virtual void state_transition(FLT dt, CnMat& cF, const CnMat& x) = 0;
        virtual void process_noise(FLT dt, const struct CnMat &x, struct CnMat &Q_out) = 0;

        virtual void predict(FLT dt, const struct CnMat &x0, struct CnMat &x1);

        virtual void sample_state(FLT dt, const struct CnMat &x0, struct CnMat &x1);

        void update(FLT t);

        void bulk_update(FLT t, const std::vector<CnMat>& Zs, const std::vector<CnMat>& Rs);
    };

    struct KalmanLinearMeasurementModel : public KalmanMeasurementModel {
        CnMat H;

        KalmanLinearMeasurementModel(KalmanModel* kalmanModel, const std::string& name, const CnMat& H);
        ~KalmanLinearMeasurementModel() override;
        bool predict_measurement(const CnMat &x, CnMat *z, CnMat *h) override;
    };
}

#endif