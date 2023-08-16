#define _USE_MATH_DEFINES
#include <cmath>

#include "matplotlibcpp.h"
#include "plot.h"

using namespace std;
namespace plt = matplotlibcpp;

void plot(const vector<double>& hist, long bins=30, string color="k", double alpha=1.0, bool density=false, bool cumulative=false, string parameter="", string units="", string save_file="plot.png") {
    string ylabel;
    if (density) { ylabel = "Density"; } else { ylabel = "Frequency"; }
    if (units != "") { units = " $("+units+")$"; }

    plt::figure_size(1000,500);
    plt::hist(hist, bins, color, alpha, density, cumulative);
    plt::xlabel(parameter+units);
    plt::ylabel(ylabel);
    plt::title(parameter+" distribution");
    plt::save(save_file,300);
}

