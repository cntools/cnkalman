#include <cnkalman/model.h>
#if HAS_SCIPLOT
#include <sciplot/sciplot.hpp>
#endif

struct ModelPlot {
#ifdef HAS_SCIPLOT
    sciplot::Plot plot;
    sciplot::Plot map;
#endif
    FLT range[4] = {INFINITY, -INFINITY, INFINITY, -INFINITY};
    bool show = false;

    std::string name;
    ModelPlot(const std::string& name = "plot");
    void plot_cov(const cnkalman::KalmanModel& model, FLT deviations, const std::string& color="red");
    void include_point_in_range(const FLT* X);
    void include_point_in_range(FLT x, FLT y);
    ~ModelPlot();
};

struct ModelRunner {
    std::string settingsName = "";
    FLT dt = 0.1;
    int iterations = 200;
    int max_iterations = 0;
    int run_every = 1;
    int ellipse_step = 20;
    bool bulk_update = false;

    ModelRunner(const std::string &settingsName = "", double dt = .1, int iterations = 200, int max_iterations = 0, int runEvery = 1, int ellipseStep = 20,
                bool bulkUpdate = false);

    void Run(cnkalman::KalmanModel& model, bool show = false, const CnMat* X = 0, const CnMat* Q = 0, std::vector<CnMat> mRs = {});
    void Run(ModelPlot& plotter, cnkalman::KalmanModel& model, const CnMat* X = 0, const CnMat* Q = 0, std::vector<CnMat> mRs = {});
};
