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
    vector<double> e(num_samples);

    mt19937 gen; // Mersenne Twister pseudo-random generator of 32-bit numbers
    Sampled_distribution<double> e_pdf(thermal, 0, 1);

    for (int i = 0; i < num_samples; i++) {
        e.at(i) = e_pdf(gen);
    }

    plot(e, 30, "blue", 1.0, true, false, "Eccentricity", "\\epsilon", "./plots/e.png");
}
