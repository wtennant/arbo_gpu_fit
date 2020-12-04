// Distribution.cpp: Defines the member functions of the Distribution class, which
// comptue the pdf of different distributions using the gsl library.

#include "gsl/gsl_randist.h"    // Random number distributions.
#include "gsl/gsl_cdf.h"        // Computing the cdf of statistical distributions.
#include "Distributions.h"      // Distributions parameter class.

// Truncated normal distribution.
// Sets the parameters of a truncated normal distribution.
void TruncatedNormal::setParams(double* in_params)
{
    params[0] = in_params[0];        // Mean.
    params[1] = in_params[1];        // Standard deviation.
    params[2] = in_params[2];        // Lower bound.
    params[3] = in_params[3];        // Upper bound.
}

// Gets the mean of a truncated normal distribution.
double TruncatedNormal::mean()
{
    return params[0];          
}

// Computes the pdf of a truncated normal distribution.
double TruncatedNormal::pdf(double x)
{
    if ((x >= params[2]) && (x <= params[3]))
    {
        return gsl_ran_gaussian_pdf(x - params[0], params[1]) / (gsl_cdf_gaussian_P(params[3] - params[0], params[1]) - gsl_cdf_gaussian_P(params[2] - params[0], params[1]));
    }
    else
    {
        return exp(-250.0);
    }    
}

// Beta distribution.
// Set the parameters for a beta distribution.
void Beta::setParams(double* in_params)
{
    params[0] = in_params[0];   // Alpha beta distribution parameter.
    params[1] = in_params[1];     // Beta beta distribution parameter.
}

// Compute the mean of a beta distribution.
double Beta::mean()
{
    return params[0] / (params[0] + params[1]);
}

// Compute the pdf of a beta distribution.
double Beta::pdf(double x)
{
    return gsl_ran_beta_pdf(x, params[0], params[1]);
}