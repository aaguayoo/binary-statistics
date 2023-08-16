#define _USE_MATH_DEFINES
#include "matplotlibcpp.h"
#include "sampled_custom_distribution.hh"
#include <cmath>
#include <iostream>
#include <fstream>

using namespace std;
namespace plt = matplotlibcpp;

double power_law(double x) {
    /*
     * PDF: 1/x -> Power law distribution
     */
    return std::log(x); // CDF
}

double thermal(double x) {
    /*
     * PDF: 2x -> Thermal distribution
     */
    return x * x; // CDF
}

double uniform(double x) {
    /*
     * PDF: 1 -> Uniform distribution
     */
    return x; // CDF
}

void plot(const vector<double>& hist, long bins=30, string color="k", double alpha=1.0, bool density=false, bool cumulative=false, string parameter="", string units="", string save_file="plot.png") {
    string ylabel;
    if (density) { ylabel = "Density"; } else { ylabel = "Counts"; }
    if (units != "") { units = " $("+units+")$"; }

    plt::figure_size(1000,500);
    plt::hist(hist, bins, color, alpha, density, cumulative);
    plt::xlabel(parameter+units);
    plt::ylabel(ylabel);
    plt::title(parameter+" distribution");
    plt::save(save_file,300);
}

int main()
{
    const int num_samples = 10000;
    vector<double> bin_sep_dist(num_samples);
    vector<double> eccen_dist(num_samples);
    vector<double> eccen2_dist(num_samples);
    vector<double> mass_dist(num_samples);
    vector<double> orbit_angle_dist(num_samples);

    mt19937 gen; // Mersenne Twister pseudo-random generator of 32-bit numbers
    Sampled_distribution<double> bin_sep_pdf(power_law, 200, 2000);
    Sampled_distribution<double> eccen_pdf(thermal, 0, 1);
    Sampled_distribution<double> eccen2_pdf(uniform, 0, 1);
    Sampled_distribution<double> mass_pdf(uniform, 0.5, 1.5);
    Sampled_distribution<double> orbit_angle_pdf(uniform, 0, 360);

    for (int i = 0; i < num_samples; i++) {
        bin_sep_dist.at(i) = bin_sep_pdf(gen);
        eccen_dist.at(i) = eccen_pdf(gen);
        eccen2_dist.at(i) = eccen2_pdf(gen);
        mass_dist.at(i) = mass_pdf(gen);
        orbit_angle_dist.at(i) = orbit_angle_pdf(gen);
    }

    plot(bin_sep_dist, 30, "orange", 1.0, true, false, "Binary separation", "UA", "./plots/binary_separation.png");
    plot(eccen_dist, 30, "blue", 1.0, true, false, "Eccentricity", "./plots/eccentricity.png");
    plot(mass_dist, 30, "green", 1.0, true, false, "Stellar mass", "M_\\odot", "./plots/stellar_mass.png");
    plot(orbit_angle_dist, 30, "red", 1.0, true, false, "Orbital angle", "^\\circ", "./plots/orbital_angle.png");
}
