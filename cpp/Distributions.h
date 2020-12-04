// Distribution.h: classes of statistical distributions, containing parameters of each
// distribution and member functions for sampling and calculating the pdf.

#include "gsl/gsl_rng.h"    // GSL random number generators.
#include <string>           // Strings.

// Start of the header guard.
#ifndef DISTRIBUTIONS_H
#define DISTRIBUTIONS_H

#define DIST_MAXPARAMS 4

// The base distribution class. Yay.
class Distribution
{
public:
    double params[DIST_MAXPARAMS];
    std::string name;
    virtual void setParams(double* in_params) = 0;
    virtual double mean() = 0;
    virtual double pdf(double x) = 0;
    virtual std::string printParams() = 0;
};

// Truncated normal distribution with params[0] = mu, params[1] = sigma, params[2] = lower, params[3] = upper.
class TruncatedNormal : public Distribution
{
public:
    void setParams(double* in_params);   // Set the parameters of the truncated normal distribution.
    double mean();                                      // Get the mean.
    double pdf(double x);                               // Compute the pdf with member parameters.
    TruncatedNormal() { name = "TruncatedNormal"; };
    
    // Member function that prints the string of parameter values.
    std::string printParams() { return "(mu = " + std::to_string(params[0]) + ";sigma = " + std::to_string(params[1])
        + ";lower = " + std::to_string(params[2]) + ";upper = " + std::to_string(params[3]) + ")"; };
};

// Beta distribution with params[0] = alpha, params[1] = beta.
class Beta : public Distribution
{
public:
    void setParams(double* in_params);    // Set the parameters.
    double mean();                                      // Get the mean.
    double pdf(double x);                               // Compute the pdf with member parameters.
    Beta() { name = "Beta"; };
    
    // Member function that prints the string of parameter values.
    std::string printParams() { return "(alpha = " + std::to_string(params[0]) + ";beta = " + std::to_string(params[1]) + ")"; }; 
};

#endif
