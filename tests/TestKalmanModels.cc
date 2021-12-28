#include "models/LinearPoseVel.h"
#include "models/EggLandscape.h"
#include "models/BikeLandmarks.h"

#include "ModelRunner.h"

#include <sciplot/sciplot.hpp>

int main() {
    BikeLandmarks bikeModel;
    EggLandscapeProblem eggModel;
    LinearPoseVel linearModel;
    RunModel(bikeModel, .1, 1, 20, false, false);
    RunModel(eggModel, 1, 1, 20, true);
    RunModel(linearModel, 1, 1, 20, false);

    return 0;
}
