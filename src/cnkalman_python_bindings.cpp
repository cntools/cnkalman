#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "cnkalman/model.h"
namespace py = pybind11;
using namespace pybind11::literals;

static inline CnMat from_py_array(py::array_t<double> b) {
    if (b.is_none()) {
        return CnMat {};
    }
    /* Request a buffer descriptor from Python */
    py::buffer_info info = b.request();

    /* Some basic validation checks ... */
    if (info.format != py::format_descriptor<double>::format())
        throw std::runtime_error("Incompatible format: expected a double array!");

    return CnMat {
            static_cast<int>(info.strides[0] / sizeof(double)),
            static_cast<double *>(info.ptr),
            static_cast<int>(info.shape[0]),
            static_cast<int>(info.ndim == 2 ? info.shape[1] : 1)
    };
}

static inline py::buffer_info to_py_array_buffer(CnMat& m) {
    if(m.cols == 1) {
        return py::buffer_info(
                m.data,                               /* Pointer to buffer */
                sizeof(double),                          /* Size of one scalar */
                py::format_descriptor<double>::format(), /* Python struct-style format descriptor */
                1,                                      /* Number of dimensions */
                { (Py_ssize_t)m.rows },                 /* Buffer dimensions */
                { (Py_ssize_t)(m.step * sizeof(double)) }
        );
    }
    return py::buffer_info(
            m.data,                               /* Pointer to buffer */
            sizeof(double),                          /* Size of one scalar */
            py::format_descriptor<double>::format(), /* Python struct-style format descriptor */
            2,                                      /* Number of dimensions */
            { (Py_ssize_t)m.rows, (Py_ssize_t)m.cols },                 /* Buffer dimensions */
            { (Py_ssize_t)(m.step * sizeof(double)),             /* Strides (in bytes) for each index */
              (Py_ssize_t)sizeof(double ) }
    );
}

static inline py::buffer_info to_py_array_buffer(const CnMat& m) {
    if(m.cols == 1) {
        return py::buffer_info(
                m.data,                               /* Pointer to buffer */
                sizeof(double),                          /* Size of one scalar */
                py::format_descriptor<double>::format(), /* Python struct-style format descriptor */
                1,                                      /* Number of dimensions */
                { (Py_ssize_t)m.rows },                 /* Buffer dimensions */
                { (Py_ssize_t)(m.step * sizeof(double)) },
                true
        );
    }
    return py::buffer_info(
            m.data,                               /* Pointer to buffer */
            sizeof(double),                          /* Size of one scalar */
            py::format_descriptor<double>::format(), /* Python struct-style format descriptor */
            2,                                      /* Number of dimensions */
            { (Py_ssize_t)m.rows, (Py_ssize_t)m.cols },                 /* Buffer dimensions */
            { (Py_ssize_t)(m.step * sizeof(double)),             /* Strides (in bytes) for each index */
              (Py_ssize_t)sizeof(double ) },
              true
    );
}

static inline py::array_t<double> to_py_array(CnMat& m) {
    return py::array_t<double>(to_py_array_buffer(m));
}

static inline py::array_t<double> to_py_array(const CnMat& m) {
    return py::array_t<double>(to_py_array_buffer(m));
}

namespace cnkalman {
    class PyKalmanMeasurementModel : public KalmanMeasurementModel {
    public:
        using KalmanMeasurementModel::KalmanMeasurementModel;
        virtual py::array_t<double> predict_measurement(py::array_t<double> x) {
            PYBIND11_OVERRIDE_PURE(
                    py::array_t<double>, /* Return type */
                    PyKalmanMeasurementModel,      /* Parent class */
                    predict_measurement,          /* Name of function in C++ (must match Python name) */
                    x     /* Argument(s) */
            );
        }

        virtual py::array_t<double> predict_measurement_jac(py::array_t<double> x) {
            PYBIND11_OVERRIDE_PURE(
                    py::array_t<double>, /* Return type */
                    PyKalmanMeasurementModel,      /* Parent class */
                    predict_measurement_jac,          /* Name of function in C++ (must match Python name) */
                    x     /* Argument(s) */
            );
        }

        bool predict_measurement(const CnMat &x, CnMat *z, CnMat *h) override {
            if(z) {
                auto _z = from_py_array(predict_measurement(to_py_array(x)));
                cn_matrix_copy(z, &_z);
            }
            if(h) {
                auto _h = from_py_array(predict_measurement_jac(to_py_array(x)));
                cn_matrix_copy(h, &_h);
            }
            return true;
        }
    };

    class PyKalmanModel : public KalmanModel {
    public:
        using KalmanModel::KalmanModel;

        virtual py::array_t<double> predict(double dt, py::array_t<double> x0) {
            PYBIND11_OVERRIDE_PURE(
                    py::array_t<double>, /* Return type */
                    PyKalmanModel,      /* Parent class */
                    predict,          /* Name of function in C++ (must match Python name) */
                    dt, x0      /* Argument(s) */
            );
        }
        virtual py::array_t<double> predict_jac(double dt, py::array_t<double> x0) {
            PYBIND11_OVERRIDE_PURE(
                    py::array_t<double>, /* Return type */
                    PyKalmanModel,      /* Parent class */
                    predict_jac,          /* Name of function in C++ (must match Python name) */
                    dt, x0      /* Argument(s) */
            );
        }

        void predict(double dt, const CnMat &x0, CnMat *x1, CnMat *cF) override {
            if(x1) {
                auto _x1 = from_py_array(predict(dt, to_py_array(x0)));
                cn_matrix_copy(x1, &_x1);
            }
            if(cF) {
                auto _f1 = from_py_array(predict_jac(dt, to_py_array(x0)));
                cn_matrix_copy(cF, &_f1);
            }
        }

        virtual py::array_t<double> process_noise(double dt, py::array_t<double> x0) {
            PYBIND11_OVERRIDE_PURE(
                    py::array_t<double>, /* Return type */
                    PyKalmanModel,      /* Parent class */
                    process_noise,          /* Name of function in C++ (must match Python name) */
                    dt, x0      /* Argument(s) */
            );
        }

        void process_noise(double dt, const CnMat &x, CnMat &Q_out) override {
            auto q = from_py_array(process_noise(dt, to_py_array(x)));
            cn_matrix_copy(&Q_out, &q);
        }
    };
}

PYBIND11_MODULE(filter, m) {
    using namespace cnkalman;

    py::class_<CnMat>(m, "CnMat", py::buffer_protocol())
            .def(py::init(&from_py_array))
            .def_buffer([](CnMat &m) -> py::buffer_info {
                return to_py_array_buffer(m);
            })
            .def_readonly("rows", &CnMat::rows)
            .def_readonly("cols", &CnMat::cols)
            ;

    py::class_<cnkalman_state_s>(m, "filter_state")
            .def_property("P", [](cnkalman_state_s& self){
                return to_py_array(self.P);
            }, [](cnkalman_state_s& self, py::array_t<double> v) {
                CnMat m = from_py_array(v);
                cn_matrix_copy(&self.P, &m);
            })
            .def_property("X", [](cnkalman_state_s& self){
                return to_py_array(self.state);
            }, [](cnkalman_state_s& self, py::array_t<double> v) {
                CnMat m = from_py_array(v);
                //cn_print_mat(&m);
                cn_matrix_copy(&self.state, &m);
            })
            .def_readwrite("state_variance_per_second", &cnkalman_state_s::state_variance_per_second)
            .def_readwrite("time", &cnkalman_state_s::t)
            ;

    py::class_<cnkalman_update_extended_total_stats_t>(m, "update_extended_total_stats")
            .def_readwrite("total_runs", &cnkalman_update_extended_total_stats_t::total_runs)
            .def_readwrite("total_failures", &cnkalman_update_extended_total_stats_t::total_failures)
            .def_readwrite("total_fevals", &cnkalman_update_extended_total_stats_t::total_fevals)
;

    py::class_<cnkalman_update_extended_stats_t>(m, "update_extended_stats")
            .def_readwrite("besterror", &cnkalman_update_extended_stats_t::besterror)
            .def_readwrite("bestnorm", &cnkalman_update_extended_stats_t::bestnorm)
            .def_readwrite("origerror", &cnkalman_update_extended_stats_t::origerror)
            .def_readwrite("stop_reason", &cnkalman_update_extended_stats_t::stop_reason)
            .def_readwrite("iterations", &cnkalman_update_extended_stats_t::iterations)
            ;

    py::class_<term_criteria_t>(m, "term_criteria_t")
            .def(py::init([](int max_iterations, double minimum_step, double xtol, double mtol, double max_error){
                return term_criteria_t {
                    max_iterations,
                    minimum_step,
                    xtol,
                    mtol,
                    max_error
                };
            }), "max_iterations"_a = 0, "minimum_step"_a=0, "xtol"_a=0, "mtol"_a=0, "max_error"_a=0)
            .def_readwrite("max_error", &term_criteria_t::max_error)
            .def_readwrite("max_iterations", &term_criteria_t::max_iterations)
            .def_readwrite("minimum_step", &term_criteria_t::minimum_step)
            .def_readwrite("mtol", &term_criteria_t::mtol)
            .def_readwrite("xtol", &term_criteria_t::xtol)
            ;

    py::enum_<enum cnkalman_jacobian_mode>(m, "jacobian_mode")
            .value("user_fn", cnkalman_jacobian_mode::cnkalman_jacobian_mode_user_fn)
            .value("debug", cnkalman_jacobian_mode::cnkalman_jacobian_mode_debug)
            .value("one_sided_minus", cnkalman_jacobian_mode::cnkalman_jacobian_mode_one_sided_minus)
            .value("one_sided_plus", cnkalman_jacobian_mode::cnkalman_jacobian_mode_one_sided_plus)
            .value("two_sided", cnkalman_jacobian_mode::cnkalman_jacobian_mode_two_sided)
            ;
    py::class_<cnkalman_meas_model>(m, "meas_state")
            .def_readwrite("adaptive", &cnkalman_meas_model::adaptive)
            .def_readwrite("meas_jacobian_mode", &cnkalman_meas_model::meas_jacobian_mode)
            .def_readwrite("term_criteria", &cnkalman_meas_model::term_criteria)
            ;

    py::class_<PyKalmanModel>(m, "Model")
            .def(py::init<const std::string&, size_t>(), "name"_a, "state_cnt"_a)
            .def(py::init<size_t>(), "state_cnt"_a)
            .def("predict", py::overload_cast<double, py::array_t<double>>(&PyKalmanModel::predict))
            .def("predict_jac", py::overload_cast<double, py::array_t<double>>(&PyKalmanModel::predict_jac))
            .def("process_noise", py::overload_cast<double, py::array_t<double>>(&PyKalmanModel::process_noise))
            .def("update", &KalmanModel::update, "time"_a)
            .def_readonly("state_cnt", &KalmanModel::state_cnt)
            .def_readonly("name", &KalmanModel::name)
            .def_readwrite("state", &KalmanModel::stateM)
            .def_readwrite("kalman_state", &KalmanModel::kalman_state)
;

    py::class_<PyKalmanMeasurementModel>(m, "MeasurementModel")
            .def(py::init<PyKalmanModel*, const std::string&, size_t>())
            .def(py::init<PyKalmanModel*, size_t>())
            .def("update", [](PyKalmanMeasurementModel& self, double t, py::array_t<double> Z,
                              py::array_t<double> R) {
                auto _Z = from_py_array(Z);
                auto _R = from_py_array(R);

                return self.update(t, _Z, _R);
            }, "time"_a, "Z"_a, "R"_a)
            .def("predict_measurement", py::overload_cast<py::array_t<double>>(&PyKalmanMeasurementModel::predict_measurement))
            .def("predict_measurement_jac", py::overload_cast<py::array_t<double>>(&PyKalmanMeasurementModel::predict_measurement_jac))
            .def("residual", &KalmanMeasurementModel::residual)
            .def("default_R", [](PyKalmanMeasurementModel& self) {
                return to_py_array(self.default_R().mat);
            })
            .def("jacobian_debug_misses", [](PyKalmanMeasurementModel& self) {
                if(self.meas_mdl.numeric_calcs == 0)
                    return 0.;
                return self.meas_mdl.numeric_misses / (double)self.meas_mdl.numeric_calcs;
            })
            .def_readwrite("meas_mdl", &KalmanMeasurementModel::meas_mdl)
;

}