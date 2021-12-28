#include <cnkalman/model.h>

struct LandmarkMeasurement : public cnkalman::KalmanMeasurementModel {
    FLT px, py;

    virtual CnMat default_R() {
        auto R = cnMatCalloc(2, 2);
        cnMatrixSet(&R, 0, 0, .3*.3);
        cnMatrixSet(&R, 1, 1, .1*.1);
        return R;
    }

    LandmarkMeasurement(cnkalman::KalmanModel *kalmanModel, double x, double y)
            : KalmanMeasurementModel(kalmanModel, "Landmark", 2), px(x), py(y) {

    }

    static inline FLT wrapAngle( FLT angle )
    {
        while (angle > M_PI)
            angle -= 2.0 * M_PI;
        while (angle < -M_PI)
            angle += 2.0 * M_PI;
        return angle;
    }

    bool residual(const CnMat &Z, const CnMat &x, CnMat *y, CnMat *h) override {
        auto rtn = KalmanMeasurementModel::residual(Z, x, y, h);
        y->data[1] = wrapAngle(y->data[1]);
        return rtn;
    }

    bool predict_measurement(const CnMat &x_t, CnMat *pz, CnMat *h) override {
        FLT x = x_t.data[0], y = x_t.data[1], theta = x_t.data[2];

        FLT hyp = pow((px - x),2) + pow((py - y),2);
        FLT dist = sqrt(hyp);

        if(pz) {
            cnMatrixSet(pz, 0, 0, dist);
            cnMatrixSet(pz, 1, 0, atan2(py - y, px - x) - theta);
        }

        if(h) {
            // Adapted to a 5 x 6....
            cnMatrixSet(h, 0, 0, (-px + x) / dist);
            cnMatrixSet(h, 0, 1, (-py + y) / dist);
            cnMatrixSet(h, 0, 2, 0);

            cnMatrixSet(h, 1, 0, -(-py + y) / hyp);
            cnMatrixSet(h, 1, 1, -(px - x) / hyp);
            cnMatrixSet(h, 1, 2, -1);
        }
        return dist > 0;
    }
};

struct BikeLandmarks : public cnkalman::KalmanModel {
    FLT wheelbase = 0.5;
    FLT v = 1.1, alpha = .01;
    FLT v_std = .1, alpha_std = .015;
    BikeLandmarks() : cnkalman::KalmanModel("BikeLandmarks", 3) {
        this->measurementModels.emplace_back(std::make_unique<LandmarkMeasurement>(this, 5, 10));
        this->measurementModels.emplace_back(std::make_unique<LandmarkMeasurement>(this, 10, 5));
        this->measurementModels.emplace_back(std::make_unique<LandmarkMeasurement>(this, 15, 15));

        state[0] = 2; state[1] = 6; state[2] = .3;
        cn_set_diag_val(&kalman_state.P, .1);
    }

    void process_noise(double dt, const struct CnMat &x, struct CnMat &Q_out) override {
        FLT d = v * dt;
        FLT tana = tan(alpha);
        if(fabs(tana) < 1e-7) {
            tana = 1e-7;
        }
        FLT c1 = (-tan(alpha)*tan(alpha)-1);
        FLT theta = x.data[2], beta = d / wheelbase * tana;
        FLT w = wheelbase;
        FLT R = wheelbase/tana;
        FLT _V[] = {
                dt * cos(beta + theta), -c1 * d * cos(beta + theta) / tana - c1 * w * sin(theta) / tana / tana + c1 * w * sin(beta + theta) / tana / tana,
                dt * sin(beta + theta), -c1 * d * sin(beta + theta) / tana + c1 * w * cos(theta) / tana / tana - c1 * w * cos(beta + theta) / tana / tana,
                dt / R, - c1 * d / w
        };
        auto V = cnMat(3, 2, _V);

        FLT _M[] = {
                v_std*v*v, 0,
                0, alpha_std*alpha_std
        };
        auto M = cnMat(2, 2, _M);
        cn_ABAt_add(&Q_out, &V, &M, 0);
    }

    void predict(double dt, const struct CnMat &x0, struct CnMat *x1, CnMat* F) override {
        FLT d = v * dt;

        FLT tana = tan(alpha);
        if(fabs(tana) < 1e-7) {
            tana = 1e-7;
        }

        FLT R, theta = x0.data[2], beta = d / wheelbase * tana;
        R = wheelbase/tana;
        if(x1) {
            FLT additional[] = {
                    -R * sin(theta) + R * sin(theta + beta),
                    R * cos(theta) - R * cos(theta + beta),
                    beta,
            };
            auto additionalM = cnVec(state_cnt, additional);
            cn_elementwise_add(x1, &x0, &additionalM);
        }

        if(F) {
            FLT f[] = {
                    1, 0, -R * cos(theta)+R*cos(theta+beta),
                    0, 1, -R * sin(theta)+R*sin(theta+beta),
                    0, 0, 1,
            };

            cn_copy_data_in(F, true, f);
        }
    }

};
