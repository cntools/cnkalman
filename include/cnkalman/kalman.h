#pragma once

#include "cnmatrix/cn_matrix.h"
#include "numerical_diff.h"


#ifdef __cplusplus
extern "C" {
#endif

/**
 * This file contains a generic kalman implementation.
 *
 * This implementation tries to use the same nomenclature as:
 * https://en.wikipedia.org/wiki/Kalman_filter#Underlying_dynamical_system_model and
 * https://en.wikipedia.org/wiki/Extended_Kalman_filter.
 *
 * This implementation supports both nonlinear prediction models and nonlinear measurement models. Each phase
 * incorporates a time delta to approximate a continous model.
 *
 * Adaptive functionality:
 *
 * https://arxiv.org/pdf/1702.00884.pdf
 *
 * The R matrix should be initialized to reasonable values on the first all and then is updated based on the residual
 * error -- higher error generates higher variance values:
 *
 * R_k = a * R_k-1 + (1 - a) * (e*e^t + H * P_k-1 * H^t)
 *
 * a is set to .3 for this implementation.

 */

struct cnkalman_state_s;

typedef void (*kalman_normalize_fn_t)(void *user, struct CnMat *x);

// Given state x0 and time delta; gives the new state x1. For a linear model, this is just x1 = F * x0 and f_out is a
// constant / time dependent
typedef void (*kalman_transition_model_fn_t)(FLT dt, const struct cnkalman_state_s *k, const struct CnMat *x0,
                                             struct CnMat *x1, struct CnMat *f_out);

// Given time and current state, generate the process noise Q_k.
typedef void (*kalman_process_noise_fn_t)(void *user, FLT dt, const struct CnMat *x, struct CnMat *Q_out);

// Given a measurement Z, and state X_t, generates both the y difference term and the H jacobian term.
typedef bool (*kalman_measurement_model_fn_t)(void *user, const struct CnMat *Z, const struct CnMat *x_t,
											  struct CnMat *y, struct CnMat *H_k);

typedef struct term_criteria_t {
	size_t max_iterations;

	// Absolute step size tolerance
	FLT minimum_step;

	// Minimum difference in errors
	FLT xtol;

	FLT mtol;
} term_criteria_t;

enum cnkalman_update_extended_termination_reason {
	cnkalman_update_extended_termination_reason_none = 0,
	cnkalman_update_extended_termination_reason_invalid_jacobian,
	cnkalman_update_extended_termination_reason_maxiter,
	cnkalman_update_extended_termination_reason_xtol,
	cnkalman_update_extended_termination_reason_step,
	cnkalman_update_extended_termination_reason_mtol,
	cnkalman_update_extended_termination_reason_MAX
};
CN_EXPORT_FUNCTION const char * cnkalman_update_extended_termination_reason_to_str(enum cnkalman_update_extended_termination_reason reason);

typedef struct cnkalman_update_extended_total_stats_t {
	FLT bestnorm_acc, orignorm_acc, bestnorm_meas_acc, bestnorm_delta_acc, orignorm_meas_acc;
	int total_iterations, total_fevals, total_hevals;
	int total_runs;
	int total_failures;
	FLT step_acc;
	int step_cnt;
	size_t stop_reason_counts[cnkalman_update_extended_termination_reason_MAX];
} cnkalman_update_extended_total_stats_t;

struct cnkalman_update_extended_stats_t {
	FLT bestnorm, bestnorm_meas, bestnorm_delta;
	FLT orignorm, orignorm_meas;
	FLT origerror, besterror;
	int iterations;
	int fevals, hevals;
	enum cnkalman_update_extended_termination_reason stop_reason;

	cnkalman_update_extended_total_stats_t *total_stats;
};

/**
 * This scheme heavily borrowed from mpfit
 */
enum cnkalman_jacobian_mode {
    cnkalman_jacobian_mode_user_fn = 0,
    cnkalman_jacobian_mode_two_sided = cnkalman_numerical_differentiate_mode_two_sided,
    cnkalman_jacobian_mode_one_sided_plus = cnkalman_numerical_differentiate_mode_one_sided_plus,
    cnkalman_jacobian_mode_one_sided_minus = cnkalman_numerical_differentiate_mode_one_sided_minus,
    cnkalman_jacobian_mode_debug = -1,
};

typedef struct cnkalman_state_s {
	// The number of states stored. For instance, something that tracked position and velocity would have 6 states --
	// [x, y, z, vx, vy, vz]
	int state_cnt;

	void *user;

    enum cnkalman_jacobian_mode transition_jacobian_mode;

	kalman_transition_model_fn_t Transition_fn;
	struct CnMat state_variance_per_second;

	kalman_process_noise_fn_t Q_fn;
	kalman_normalize_fn_t normalize_fn;

	// Store the current covariance matrix (state_cnt x state_cnt)
	struct CnMat P;

	// Actual state matrix and whether its stored on the heap. Make no assumptions about how this matrix is organized.
	// it is always size of state_cnt*sizeof(FLT) though.
	bool State_is_heap;
	struct CnMat state;

	// Current time
	FLT t;

	int log_level;
	void *datalog_user;
	void (*datalog)(struct cnkalman_state_s *state, const char *name, const FLT *v, size_t length);
} cnkalman_state_t;

typedef struct cnkalman_meas_model {
	cnkalman_state_t *k;
    enum cnkalman_jacobian_mode meas_jacobian_mode;

	const char *name;
	kalman_measurement_model_fn_t Hfn;

	bool adaptive;

	struct term_criteria_t term_criteria;
	cnkalman_update_extended_total_stats_t stats;
} cnkalman_meas_model_t;

CN_EXPORT_FUNCTION FLT cnkalman_meas_model_predict_update_stats(FLT t, cnkalman_meas_model_t *mk, void *user,
																  const struct CnMat *Z, CnMat *R,
																  struct cnkalman_update_extended_stats_t *stats);
CN_EXPORT_FUNCTION FLT cnkalman_meas_model_predict_update(FLT t, cnkalman_meas_model_t *mk, void *user,
															const struct CnMat *Z, CnMat *R);

/**
 * Predict the state at a given delta; doesn't update the covariance matrix
 * @param t delta time
 * @param k kalman state info
 * @param index Which state vector to pull out
 * @param out Pre allocated output buffer.
 */
CN_EXPORT_FUNCTION void cnkalman_extrapolate_state(FLT t, const cnkalman_state_t *k, size_t start_index,
                                                         size_t end_index, FLT *out);

CN_EXPORT_FUNCTION void cnkalman_predict_state(FLT t, cnkalman_state_t *k);

/**
 * Run predict and update, updating the state matrix. This is for purely linear measurement models.
 *
 * @param t absolute time
 * @param k kalman state info
 * @param z measurement -- CnMat of n x 1
 * @param H Input observation model -- CnMat of n x state_cnt
 * @param R Observation noise -- The diagonal of the measurement covariance matrix; length n
 * @param adapative Whether or not R is an adaptive matrix. When true, R should be a full n x n matrix.
 *
 */
CN_EXPORT_FUNCTION FLT cnkalman_predict_update_state(FLT t, cnkalman_state_t *k, const struct CnMat *Z,
													   const struct CnMat *H, CnMat *R, bool adaptive);

/**
 * Run predict and update, updating the state matrix. This is for non-linear measurement models.
 *
 * @param t absolute time
 * @param k kalman state info
 * @param z measurement -- CnMat of n x 1
 * @param R Observation noise -- The diagonal of the measurement covariance matrix; length n
 * @param extended_params parameters for the non linear update
 * @param stats store stats if requested
 *
 * @returns Returns the average residual error
 */
/*
CN_EXPORT_FUNCTION FLT
cnkalman_predict_update_state_extended(FLT t, cnkalman_state_t *k, const struct CnMat *Z, CnMat* R,
											 const cnkalman_update_extended_params_t *extended_params,
											 struct cnkalman_update_extended_stats_t *stats);
*/
/**
 * Initialize a kalman state object
 * @param k object to initialize
 * @param state_cnt Length of state vector
 * @param F Transition function
 * @param q_fn Noise function
 * @param user pointer to give to user functions
 * @param state Optional state. Pass 0 to malloc one. Otherwise should point to a vector of at least state_cnt FLTs.
 *
 * @returns Returns the average residual error
 */
CN_EXPORT_FUNCTION void cnkalman_state_init(cnkalman_state_t *k, size_t state_cnt, kalman_transition_model_fn_t F,
											  kalman_process_noise_fn_t q_fn, void *user, FLT *state);

CN_EXPORT_FUNCTION void cnkalman_meas_model_init(cnkalman_state_t *k, const char *name,
												   cnkalman_meas_model_t *mk, kalman_measurement_model_fn_t Hfn);
CN_EXPORT_FUNCTION void cnkalman_state_reset(cnkalman_state_t *k);

CN_EXPORT_FUNCTION void cnkalman_state_free(cnkalman_state_t *k);
CN_EXPORT_FUNCTION void cnkalman_set_P(cnkalman_state_t *k, const FLT *d);
CN_EXPORT_FUNCTION void cnkalman_set_logging_level(cnkalman_state_t *k, int verbosity);

CN_EXPORT_FUNCTION void cnkalman_linear_update(struct CnMat *F, const struct CnMat *x0, struct CnMat *x1);

#ifdef __cplusplus
}
#endif
