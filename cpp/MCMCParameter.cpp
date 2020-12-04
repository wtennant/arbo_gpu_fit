#include "MCMCParameter.h"

MCMCParameter<TruncatedNormal>::MCMCParameter(std::string _name, double _mu, double _sigma, double _lower, double _upper)
{
    name = _name;
    double params[DIST_MAXPARAMS]{ _mu, _sigma, _lower, _upper };
    prior.setParams(params);
}

MCMCParameter<Beta>::MCMCParameter(std::string _name, double _alpha, double _beta)
{
    name = _name;
    double params[DIST_MAXPARAMS]{ _alpha, _beta };
    prior.setParams(params);
}

MCMCParameterPointer::MCMCParameterPointer(MCMCParameter<TruncatedNormal>* _MCMCParameter)
{
    name = &_MCMCParameter->name;
    accepted = &_MCMCParameter->accepted;
    candidate = &_MCMCParameter->candidate;
    prior = &_MCMCParameter->prior;
}

MCMCParameterPointer::MCMCParameterPointer(MCMCParameter<Beta>* _MCMCParameter)
{
    name = &_MCMCParameter->name;
    accepted = &_MCMCParameter->accepted;
    candidate = &_MCMCParameter->candidate;
    prior = &_MCMCParameter->prior;
}