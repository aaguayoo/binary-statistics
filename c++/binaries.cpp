#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <fstream>

#include "distributions.h"
#include "matplotlibcpp.h"
#include "plot.h"
#include "sampled_custom_distribution.hh"


using namespace std;

int main()
{
    const int num_samples = 10000;
    vector<double> bin_sep_dist(num_samples);
    vector<double> eccen_dist(num_samples);
    vector<double> eccen2_dist(num_samples);
    vector<double> mass_dist(num_samples);
    vector<double> orbit_angle_dist(num_samples);
    vector<double> orbit_phase_dist(num_samples);
    vector<double> separation_dist(num_samples);

    mt19937 gen; // Mersenne Twister pseudo-random generator of 32-bit numbers
    Sampled_distribution<double> bin_sep_pdf(power_law, 200, 2000);
    Sampled_distribution<double> eccen_pdf(thermal, 0, 1);
    Sampled_distribution<double> eccen2_pdf(uniform, 0, 1);
    Sampled_distribution<double> mass_pdf(uniform, 0.5, 1.5);
    Sampled_distribution<double> orbit_angle_pdf(uniform, 0, 2*M_PI);
    Sampled_distribution<double> orbit_phase_pdf(uniform, -M_PI+0.1, M_PI-0.1);

    for (int i = 0; i < num_samples; i++) {
        bin_sep_dist.at(i) = bin_sep_pdf(gen);
        eccen_dist.at(i) = eccen_pdf(gen);
        eccen2_dist.at(i) = eccen2_pdf(gen);
        mass_dist.at(i) = mass_pdf(gen);
        orbit_angle_dist.at(i) = orbit_angle_pdf(gen);
        orbit_phase_dist.at(i) = orbit_phase_pdf(gen);
        separation_dist.at(i) = bin_sep_dist.at(i)*(1.0 - eccen2_dist.at(i))/(1.0 + eccen_dist.at(i)*std::cos(orbit_phase_dist.at(i)));
    }

    plot(bin_sep_dist, 30, "orange", 1.0, false, false, "Orbit semi-axis", "UA", "./plots/orbit_semi.png");
    plot(eccen_dist, 30, "blue", 1.0, false, false, "Eccentricity", "\\epsilon", "./plots/eccentricity.png");
    plot(eccen2_dist, 30, "blue", 1.0, false, false, "Eccentricity squared", "\\epsilon^2","./plots/eccentricity2.png");
    plot(mass_dist, 30, "green", 1.0, false, false, "Stellar mass", "M_\\odot", "./plots/stellar_mass.png");
    plot(orbit_angle_dist, 30, "red", 1.0, false, false, "Orbital angle", "rad", "./plots/orbital_angle.png");
    plot(orbit_phase_dist, 30, "purple", 1.0, false, false, "Orbital phase", "rad", "./plots/orbital_phase.png");
    plot(separation_dist, 300, "gray", 1.0, false, false, "Binary separation", "UA", "./plots/separation.png");
}
