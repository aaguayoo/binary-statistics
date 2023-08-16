#define _USE_MATH_DEFINES
#include "sampled_custom_distribution.hh"
#include "distributions.h"
#include <cmath>

using namespace std;

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
