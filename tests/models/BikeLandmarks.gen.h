static inline void gen_predict_function(CnMat* out, const FLT dt, const FLT wheelbase, const FLT* state, const FLT* u) {
	const FLT state0 = state[0];
	const FLT state1 = state[1];
	const FLT state2 = state[2];
	const FLT u0 = u[0];
	const FLT u1 = u[1];
	const FLT x0 = tan(u1);
	const FLT x1 = (1. / x0) * wheelbase;
	const FLT x2 = state2 + (u0 * x0 * dt * (1. / wheelbase));
	cnMatrixSet(out, 0, 0, (x1 * sin(x2)) + state0 + (-1 * x1 * sin(state2)));
	cnMatrixSet(out, 1, 0, (x1 * cos(state2)) + state1 + (-1 * x1 * cos(x2)));
	cnMatrixSet(out, 2, 0, x2);
}

// Jacobian of predict_function wrt [dt]
static inline void gen_predict_function_jac_dt(CnMat* Hx, const FLT dt, const FLT wheelbase, const FLT* state, const FLT* u) {
	const FLT state2 = state[2];
	const FLT u0 = u[0];
	const FLT u1 = u[1];
	const FLT x0 = u0 * tan(u1) * (1. / wheelbase);
	const FLT x1 = state2 + (x0 * dt);
	cnMatrixSet(Hx, 0, 0, u0 * cos(x1));
	cnMatrixSet(Hx, 1, 0, u0 * sin(x1));
	cnMatrixSet(Hx, 2, 0, x0);
}

// Full version Jacobian of predict_function wrt [dt]
static inline void gen_predict_function_jac_dt_with_hx(CnMat* Hx, CnMat* hx, const FLT dt, const FLT wheelbase, const FLT* state, const FLT* u) {
    if(Hx == 0) { 
        gen_predict_function(hx, dt, wheelbase, state, u);
        return;
    }
    if(hx == 0) { 
        gen_predict_function_jac_dt(Hx, dt, wheelbase, state, u);
        return;
    }
	const FLT state0 = state[0];
	const FLT state1 = state[1];
	const FLT state2 = state[2];
	const FLT u0 = u[0];
	const FLT u1 = u[1];
	const FLT x0 = tan(u1);
	const FLT x1 = u0 * x0 * (1. / wheelbase);
	const FLT x2 = state2 + (x1 * dt);
	const FLT x3 = cos(x2);
	const FLT x4 = sin(x2);
	const FLT x5 = (1. / x0) * wheelbase;
	cnMatrixSet(Hx, 0, 0, u0 * x3);
	cnMatrixSet(Hx, 1, 0, u0 * x4);
	cnMatrixSet(Hx, 2, 0, x1);
	cnMatrixSet(hx, 0, 0, (x4 * x5) + state0 + (-1 * x5 * sin(state2)));
	cnMatrixSet(hx, 1, 0, (x5 * cos(state2)) + state1 + (-1 * x3 * x5));
	cnMatrixSet(hx, 2, 0, x2);
}

// Jacobian of predict_function wrt [wheelbase]
static inline void gen_predict_function_jac_wheelbase(CnMat* Hx, const FLT dt, const FLT wheelbase, const FLT* state, const FLT* u) {
	const FLT state2 = state[2];
	const FLT u0 = u[0];
	const FLT u1 = u[1];
	const FLT x0 = tan(u1);
	const FLT x1 = 1. / x0;
	const FLT x2 = u0 * dt;
	const FLT x3 = x2 * (1. / wheelbase);
	const FLT x4 = state2 + (x0 * x3);
	const FLT x5 = sin(x4);
	const FLT x6 = cos(x4);
	cnMatrixSet(Hx, 0, 0, (-1 * x3 * x6) + (x1 * x5) + (-1 * x1 * sin(state2)));
	cnMatrixSet(Hx, 1, 0, (-1 * x1 * x6) + (x1 * cos(state2)) + (-1 * x3 * x5));
	cnMatrixSet(Hx, 2, 0, -1 * x0 * x2 * (1. / (wheelbase * wheelbase)));
}

// Full version Jacobian of predict_function wrt [wheelbase]
static inline void gen_predict_function_jac_wheelbase_with_hx(CnMat* Hx, CnMat* hx, const FLT dt, const FLT wheelbase, const FLT* state, const FLT* u) {
    if(Hx == 0) { 
        gen_predict_function(hx, dt, wheelbase, state, u);
        return;
    }
    if(hx == 0) { 
        gen_predict_function_jac_wheelbase(Hx, dt, wheelbase, state, u);
        return;
    }
	const FLT state0 = state[0];
	const FLT state1 = state[1];
	const FLT state2 = state[2];
	const FLT u0 = u[0];
	const FLT u1 = u[1];
	const FLT x0 = tan(u1);
	const FLT x1 = 1. / x0;
	const FLT x2 = u0 * dt;
	const FLT x3 = x2 * (1. / wheelbase);
	const FLT x4 = state2 + (x0 * x3);
	const FLT x5 = sin(x4);
	const FLT x6 = x1 * x5;
	const FLT x7 = x1 * sin(state2);
	const FLT x8 = cos(x4);
	const FLT x9 = x1 * cos(state2);
	const FLT x10 = x1 * x8;
	cnMatrixSet(Hx, 0, 0, (-1 * x3 * x8) + x6 + (-1 * x7));
	cnMatrixSet(Hx, 1, 0, (-1 * x10) + x9 + (-1 * x3 * x5));
	cnMatrixSet(Hx, 2, 0, -1 * x0 * x2 * (1. / (wheelbase * wheelbase)));
	cnMatrixSet(hx, 0, 0, (x6 * wheelbase) + state0 + (-1 * x7 * wheelbase));
	cnMatrixSet(hx, 1, 0, (x9 * wheelbase) + state1 + (-1 * x10 * wheelbase));
	cnMatrixSet(hx, 2, 0, x4);
}

// Jacobian of predict_function wrt [state0, state1, state2]
static inline void gen_predict_function_jac_state(CnMat* Hx, const FLT dt, const FLT wheelbase, const FLT* state, const FLT* u) {
	const FLT state2 = state[2];
	const FLT u0 = u[0];
	const FLT u1 = u[1];
	const FLT x0 = tan(u1);
	const FLT x1 = state2 + (u0 * x0 * dt * (1. / wheelbase));
	const FLT x2 = (1. / x0) * wheelbase;
	cnMatrixSet(Hx, 0, 0, 1);
	cnMatrixSet(Hx, 0, 1, 0);
	cnMatrixSet(Hx, 0, 2, (-1 * x2 * cos(state2)) + (x2 * cos(x1)));
	cnMatrixSet(Hx, 1, 0, 0);
	cnMatrixSet(Hx, 1, 1, 1);
	cnMatrixSet(Hx, 1, 2, (x2 * sin(x1)) + (-1 * x2 * sin(state2)));
	cnMatrixSet(Hx, 2, 0, 0);
	cnMatrixSet(Hx, 2, 1, 0);
	cnMatrixSet(Hx, 2, 2, 1);
}

// Full version Jacobian of predict_function wrt [state0, state1, state2]
static inline void gen_predict_function_jac_state_with_hx(CnMat* Hx, CnMat* hx, const FLT dt, const FLT wheelbase, const FLT* state, const FLT* u) {
    if(Hx == 0) { 
        gen_predict_function(hx, dt, wheelbase, state, u);
        return;
    }
    if(hx == 0) { 
        gen_predict_function_jac_state(Hx, dt, wheelbase, state, u);
        return;
    }
	const FLT state0 = state[0];
	const FLT state1 = state[1];
	const FLT state2 = state[2];
	const FLT u0 = u[0];
	const FLT u1 = u[1];
	const FLT x0 = tan(u1);
	const FLT x1 = state2 + (u0 * x0 * dt * (1. / wheelbase));
	const FLT x2 = (1. / x0) * wheelbase;
	const FLT x3 = x2 * cos(x1);
	const FLT x4 = x2 * cos(state2);
	const FLT x5 = (x2 * sin(x1)) + (-1 * x2 * sin(state2));
	cnMatrixSet(Hx, 0, 0, 1);
	cnMatrixSet(Hx, 0, 1, 0);
	cnMatrixSet(Hx, 0, 2, (-1 * x4) + x3);
	cnMatrixSet(Hx, 1, 0, 0);
	cnMatrixSet(Hx, 1, 1, 1);
	cnMatrixSet(Hx, 1, 2, x5);
	cnMatrixSet(Hx, 2, 0, 0);
	cnMatrixSet(Hx, 2, 1, 0);
	cnMatrixSet(Hx, 2, 2, 1);
	cnMatrixSet(hx, 0, 0, x5 + state0);
	cnMatrixSet(hx, 1, 0, x4 + state1 + (-1 * x3));
	cnMatrixSet(hx, 2, 0, x1);
}

// Jacobian of predict_function wrt [u0, u1]
static inline void gen_predict_function_jac_u(CnMat* Hx, const FLT dt, const FLT wheelbase, const FLT* state, const FLT* u) {
	const FLT state2 = state[2];
	const FLT u0 = u[0];
	const FLT u1 = u[1];
	const FLT x0 = tan(u1);
	const FLT x1 = dt * (1. / wheelbase);
	const FLT x2 = x0 * x1;
	const FLT x3 = state2 + (u0 * x2);
	const FLT x4 = cos(x3);
	const FLT x5 = x4 * dt;
	const FLT x6 = x0 * x0;
	const FLT x7 = 1 + x6;
	const FLT x8 = (1. / x6) * x7 * wheelbase;
	const FLT x9 = sin(x3);
	const FLT x10 = u0 * x7;
	const FLT x11 = (1. / x0) * x10;
	const FLT x12 = x9 * dt;
	cnMatrixSet(Hx, 0, 0, x5);
	cnMatrixSet(Hx, 0, 1, (x5 * x11) + (x8 * sin(state2)) + (-1 * x8 * x9));
	cnMatrixSet(Hx, 1, 0, x12);
	cnMatrixSet(Hx, 1, 1, (x4 * x8) + (-1 * x8 * cos(state2)) + (x12 * x11));
	cnMatrixSet(Hx, 2, 0, x2);
	cnMatrixSet(Hx, 2, 1, x1 * x10);
}

// Full version Jacobian of predict_function wrt [u0, u1]
static inline void gen_predict_function_jac_u_with_hx(CnMat* Hx, CnMat* hx, const FLT dt, const FLT wheelbase, const FLT* state, const FLT* u) {
    if(Hx == 0) { 
        gen_predict_function(hx, dt, wheelbase, state, u);
        return;
    }
    if(hx == 0) { 
        gen_predict_function_jac_u(Hx, dt, wheelbase, state, u);
        return;
    }
	const FLT state0 = state[0];
	const FLT state1 = state[1];
	const FLT state2 = state[2];
	const FLT u0 = u[0];
	const FLT u1 = u[1];
	const FLT x0 = tan(u1);
	const FLT x1 = dt * (1. / wheelbase);
	const FLT x2 = x0 * x1;
	const FLT x3 = state2 + (u0 * x2);
	const FLT x4 = cos(x3);
	const FLT x5 = x4 * dt;
	const FLT x6 = sin(state2);
	const FLT x7 = x0 * x0;
	const FLT x8 = 1 + x7;
	const FLT x9 = x8 * (1. / x7) * wheelbase;
	const FLT x10 = sin(x3);
	const FLT x11 = 1. / x0;
	const FLT x12 = u0 * x8;
	const FLT x13 = x12 * x11;
	const FLT x14 = dt * x10;
	const FLT x15 = cos(state2);
	const FLT x16 = x11 * wheelbase;
	cnMatrixSet(Hx, 0, 0, x5);
	cnMatrixSet(Hx, 0, 1, (x5 * x13) + (x6 * x9) + (-1 * x9 * x10));
	cnMatrixSet(Hx, 1, 0, x14);
	cnMatrixSet(Hx, 1, 1, (x4 * x9) + (-1 * x9 * x15) + (x14 * x13));
	cnMatrixSet(Hx, 2, 0, x2);
	cnMatrixSet(Hx, 2, 1, x1 * x12);
	cnMatrixSet(hx, 0, 0, (x10 * x16) + state0 + (-1 * x6 * x16));
	cnMatrixSet(hx, 1, 0, (x15 * x16) + state1 + (-1 * x4 * x16));
	cnMatrixSet(hx, 2, 0, x3);
}

static inline void gen_meas_function(CnMat* out, const FLT* state, const FLT* landmark) {
	const FLT state0 = state[0];
	const FLT state1 = state[1];
	const FLT state2 = state[2];
	const FLT landmark0 = landmark[0];
	const FLT landmark1 = landmark[1];
	const FLT x0 = landmark1 + (-1 * state1);
	const FLT x1 = landmark0 + (-1 * state0);
	cnMatrixSet(out, 0, 0, sqrt((x1 * x1) + (x0 * x0)));
	cnMatrixSet(out, 1, 0, atan2(x0, x1) + (-1 * state2));
}

// Jacobian of meas_function wrt [state0, state1, state2]
static inline void gen_meas_function_jac_state(CnMat* Hx, const FLT* state, const FLT* landmark) {
	const FLT state0 = state[0];
	const FLT state1 = state[1];
	const FLT landmark0 = landmark[0];
	const FLT landmark1 = landmark[1];
	const FLT x0 = landmark1 + (-1 * state1);
	const FLT x1 = landmark0 + (-1 * state0);
	const FLT x2 = (x1 * x1) + (x0 * x0);
	const FLT x3 = 1. / sqrt(x2);
	const FLT x4 = 1. / x2;
	cnMatrixSet(Hx, 0, 0, -1 * x1 * x3);
	cnMatrixSet(Hx, 0, 1, -1 * x0 * x3);
	cnMatrixSet(Hx, 0, 2, 0);
	cnMatrixSet(Hx, 1, 0, x0 * x4);
	cnMatrixSet(Hx, 1, 1, -1 * x1 * x4);
	cnMatrixSet(Hx, 1, 2, -1);
}

// Full version Jacobian of meas_function wrt [state0, state1, state2]
static inline void gen_meas_function_jac_state_with_hx(CnMat* Hx, CnMat* hx, const FLT* state, const FLT* landmark) {
    if(Hx == 0) { 
        gen_meas_function(hx, state, landmark);
        return;
    }
    if(hx == 0) { 
        gen_meas_function_jac_state(Hx, state, landmark);
        return;
    }
	const FLT state0 = state[0];
	const FLT state1 = state[1];
	const FLT state2 = state[2];
	const FLT landmark0 = landmark[0];
	const FLT landmark1 = landmark[1];
	const FLT x0 = landmark1 + (-1 * state1);
	const FLT x1 = landmark0 + (-1 * state0);
	const FLT x2 = (x1 * x1) + (x0 * x0);
	const FLT x3 = sqrt(x2);
	const FLT x4 = 1. / x3;
	const FLT x5 = 1. / x2;
	cnMatrixSet(Hx, 0, 0, -1 * x1 * x4);
	cnMatrixSet(Hx, 0, 1, -1 * x0 * x4);
	cnMatrixSet(Hx, 0, 2, 0);
	cnMatrixSet(Hx, 1, 0, x0 * x5);
	cnMatrixSet(Hx, 1, 1, -1 * x1 * x5);
	cnMatrixSet(Hx, 1, 2, -1);
	cnMatrixSet(hx, 0, 0, x3);
	cnMatrixSet(hx, 1, 0, atan2(x0, x1) + (-1 * state2));
}

// Jacobian of meas_function wrt [landmark0, landmark1]
static inline void gen_meas_function_jac_landmark(CnMat* Hx, const FLT* state, const FLT* landmark) {
	const FLT state0 = state[0];
	const FLT state1 = state[1];
	const FLT landmark0 = landmark[0];
	const FLT landmark1 = landmark[1];
	const FLT x0 = landmark1 + (-1 * state1);
	const FLT x1 = landmark0 + (-1 * state0);
	const FLT x2 = (x1 * x1) + (x0 * x0);
	const FLT x3 = 1. / sqrt(x2);
	const FLT x4 = 1. / x2;
	cnMatrixSet(Hx, 0, 0, x1 * x3);
	cnMatrixSet(Hx, 0, 1, x0 * x3);
	cnMatrixSet(Hx, 1, 0, -1 * x0 * x4);
	cnMatrixSet(Hx, 1, 1, x1 * x4);
}

// Full version Jacobian of meas_function wrt [landmark0, landmark1]
static inline void gen_meas_function_jac_landmark_with_hx(CnMat* Hx, CnMat* hx, const FLT* state, const FLT* landmark) {
    if(Hx == 0) { 
        gen_meas_function(hx, state, landmark);
        return;
    }
    if(hx == 0) { 
        gen_meas_function_jac_landmark(Hx, state, landmark);
        return;
    }
	const FLT state0 = state[0];
	const FLT state1 = state[1];
	const FLT state2 = state[2];
	const FLT landmark0 = landmark[0];
	const FLT landmark1 = landmark[1];
	const FLT x0 = landmark1 + (-1 * state1);
	const FLT x1 = landmark0 + (-1 * state0);
	const FLT x2 = (x1 * x1) + (x0 * x0);
	const FLT x3 = sqrt(x2);
	const FLT x4 = 1. / x3;
	const FLT x5 = 1. / x2;
	cnMatrixSet(Hx, 0, 0, x1 * x4);
	cnMatrixSet(Hx, 0, 1, x0 * x4);
	cnMatrixSet(Hx, 1, 0, -1 * x0 * x5);
	cnMatrixSet(Hx, 1, 1, x1 * x5);
	cnMatrixSet(hx, 0, 0, x3);
	cnMatrixSet(hx, 1, 0, atan2(x0, x1) + (-1 * state2));
}

