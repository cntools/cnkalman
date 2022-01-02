#include <models/BearingsOnlyTracking.h>
#include "models/LinearPoseVel.h"
#include "models/EggLandscape.h"
#include "models/BikeLandmarks.h"

#include "ModelRunner.h"

int main() {
    ModelRunner defaultRunner;
    ModelRunner defaultRunnerEvery10;
    defaultRunnerEvery10.run_every = 10;

    BikeLandmarks bikeModel;
    EggLandscapeProblem eggModel;
    LinearPoseVel linearModel;
    BearingsOnlyTracking bearingsOnlyTracking;

    defaultRunnerEvery10.Run(bikeModel);
    defaultRunner.Run(eggModel);
    defaultRunner.Run(linearModel);
    defaultRunner.Run(bearingsOnlyTracking);

    CN_CREATE_STACK_MAT(Q, 2, 2);
    std::vector<CnMat> Rs;
    Rs.emplace_back(cnMatCalloc(1, 1));
    Rs.emplace_back(cnMatCalloc(1, 1));
    Rs.emplace_back(cnMatCalloc(1, 1));

    CN_CREATE_STACK_VEC(initial_x, 2);
    initial_x.data[0] = .5; initial_x.data[1] = .1;

    ModelPlot iekfPlot("IEKF vs not");
    ModelRunner testRunner("small-iteration", .1, 5, 0, 1, 1, false);
    testRunner.Run(iekfPlot, bearingsOnlyTracking, &initial_x, &Q, Rs);

    ModelRunner testRunnerIEKF("small-iteration-iekf", .1, 5, 5, 1, 1, false);
    testRunnerIEKF.Run(iekfPlot, bearingsOnlyTracking, &initial_x, &Q, Rs);

    return 0;
}
