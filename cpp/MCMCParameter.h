// MCMCParameter.h: template class for unobserved parameters.
#include <string>
#include "Distributions.h"

// Start  of header guard.
#ifndef MCMCPARAMETER_H
#define MCMCPARAMETER_H

// Template class takes a prior and a proposal distribution.
template<class Prior> 
class MCMCParameter
{
public:
    std::string name;   // The name of the unobserved parameter as a string.
    double* accepted;         // The address of the chain of the unobserved parameter.
    double candidate;         // The proposal value for the unobserved parameter.
    Prior prior;              // The prior distribution of the unobserved parameter.
    MCMCParameter(std::string _name, double _mu, double _sigma, double _lower, double _upper);
    MCMCParameter(std::string _name, double _alpha, double _beta);
};

class MCMCParameterPointer
{
public:
    std::string* name;
    double** accepted;
    double* candidate;
    Distribution* prior;
    MCMCParameterPointer(){};
    MCMCParameterPointer(MCMCParameter<TruncatedNormal>* _MCMCParameter);
    MCMCParameterPointer(MCMCParameter<Beta>* _MCMCParameter);
};

#endif