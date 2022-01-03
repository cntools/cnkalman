/// NOTE: This is a generated file; do not edit.
#pragma once
#include <cnkalman/generated_header.h>
// clang-format off    
static inline void gen_predict_meas(CnMat* out, const FLT* x) {
	const FLT x0 = x[0];
	const FLT x1 = x[1];
	cnSetZero(out);
	cnMatrixSet(out, 0, 0, sin((5 * (x1 * x1)) + (5 * (x0 * x0))));
	cnMatrixSet(out, 1, 0, sin(101 * x0) * sin(101 * x1));
	cnMatrixSet(out, 2, 0, cos(11 * x1) * sin(11 * x0));
}

// Jacobian of predict_meas wrt [x0, x1, x2, x3]
static inline void gen_predict_meas_jac_x(CnMat* Hx, const FLT* x) {
	const FLT x0 = x[0];
	const FLT x1 = x[1];
	const FLT x2 = 10 * cos((5 * (x1 * x1)) + (5 * (x0 * x0)));
	const FLT x3 = 101 * x1;
	const FLT x4 = 101 * x0;
	const FLT x5 = 11 * x1;
	const FLT x6 = 11 * x0;
	cnSetZero(Hx);
	cnMatrixSet(Hx, 0, 0, x0 * x2);
	cnMatrixSet(Hx, 0, 1, x2 * x1);
	cnMatrixSet(Hx, 1, 0, 101 * sin(x3) * cos(x4));
	cnMatrixSet(Hx, 1, 1, 101 * sin(x4) * cos(x3));
	cnMatrixSet(Hx, 2, 0, 11 * cos(x6) * cos(x5));
	cnMatrixSet(Hx, 2, 1, -11 * sin(x6) * sin(x5));
}

// Full version Jacobian of predict_meas wrt [x0, x1, x2, x3]

static inline void gen_predict_meas_jac_x_with_hx(CnMat* Hx, CnMat* hx, const FLT* x) {
    if(hx != 0) { 
        gen_predict_meas(hx, x);
    }
    if(Hx != 0) { 
        gen_predict_meas_jac_x(Hx, x);
    }
}
