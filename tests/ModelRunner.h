#include <cnkalman/model.h>

void RunModel(cnkalman::KalmanModel& model, FLT dt = 1, int run_every = 1, int ellipse_step = 20, bool show = false, bool bulk_update = false);