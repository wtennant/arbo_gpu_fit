// MCMC.h: defines the class for variables involved in MCMC, including
// the chains for each unobserved parameter, and the prior and proposal distribution for each
// unobserved parameter.

#include <cstdint>              // Fixed-width integers.
#include "gsl\gsl_vector.h"
#include "gsl\gsl_matrix.h"
#include "gsl\gsl_rng.h"        // Random number generators.
#include "IOParam.h"            // Input-output paths.
#include "SimulationParam.h"    // Simulation parameter class.
#include "DengueData.h"         // Dengue data class.
#include "MCMCParameter.h"      // MCMC parameter class.
#include "Distributions.h"      // Statistical distribution classes.

// Start of the header guard.
#ifndef MCMC_H
#define MCMC_H

class MCMC
{
private:
    // Variables used throughout the MCMC.
    uint32_t maxTime;                                       // Maximum time (days) of each simulation.
    uint32_t maxWeek;                                       // Maximum number of weeks in the data.
    uint32_t maxStep;                                       // Maximum number of steps in the MCMC chain.
    double logLikelihood;                                   // The log-likelihood of the current step in the MCMC.
    double prevAcceptedLogLikelihood;                       // The log-likelihood of the previously accepted set of parameters.
    double acceptanceRate;                                  // Current acceptance rate of MCMC chain.
    double* particleLogLikelihood;                          // The current log-likelihood of each simulated epidemic curve.
    uint32_t* weekSimIncidence;                             // Weekly incidence from each simulation of the same parameter set.
    gsl_matrix* chainCov;                                   // Approximation of the covariance matrix of the MCMC parameters.    
    gsl_rng* rng;                                           // RNG used in sampling from proposal distributions.
    MCMCParameterPointer* MCMCParam;

    // The unobserved parameters approximated by the MCMC.
    const uint32_t nParams = 7;                                                     // Number of MCMC Parameters.
    MCMCParameter<TruncatedNormal> scaleEIP =                                       // Linear scalar for 1 / extrinsic incubation period.
        MCMCParameter<TruncatedNormal>("scaleEIP", 2.5, 0.5, 0, 10000);             
    MCMCParameter<TruncatedNormal> scaleMLE =                                       // Linear scalar for 1 / mosquito death rate.
        MCMCParameter<TruncatedNormal>("scaleMLE", 3, 0.5, 0, 10000);               
    MCMCParameter<TruncatedNormal> scaleCC =                                        // Non-linear scalar for seasonality of rainfall on mosquito carrying capacity.
        MCMCParameter<TruncatedNormal>("scaleCC", 1, 0.4, 0, 10000);                
    MCMCParameter<TruncatedNormal> scaleHumidity =                                  // Non-linear scalar for effect of humidity on mosquito mortality rate.
        MCMCParameter<TruncatedNormal>("scaleHumidity", 1, 0.4, 0, 10000);   
    MCMCParameter<TruncatedNormal> minMosToHuman =                                  // Minimum mosquito to human ratio.
        MCMCParameter<TruncatedNormal>("minMosToHuman", 3, 1, 0, 10000);
    MCMCParameter<Beta> dispersion =                                                // Dispersion parameter for the negative binomial distribution in likelihood.
        MCMCParameter<Beta>("dispersion", 20, 80);
    MCMCParameter<Beta> observationRate =                                           // Proportion of cases observed in the data.
        MCMCParameter<Beta>("observationRate", 10, 90);

    //
    void calcEachLogLikelihood(DengueData* h_dengueData, SimulationParam* h_simParam);         // Calculates the log-likelihood of EACH point in the real data given parameters and simulated data.

public:
    MCMC(uint32_t input_maxSteps, uint32_t input_maxTime, uint32_t input_maxWeek);  // Constructor for MCMC class.
    ~MCMC();                                                                        // Deconstructor for MCMC class.
    
    // Doing the MCMC in order of the following member functions.
    void calcCov(uint32_t currentStep);                                                 // Calculate the approximation to the covariance matrix of the parameters.
    void propose(uint32_t currentStep);                                                 // Determine a set of proposal values for each MCMC parameter.
    bool checkParams();                                                                 // Check that the candidate parameters make sense.
    void copyParams(SimulationParam* h_simParam, ClimateData* h_climateData);           // Set the parameters in the simulation to proposal parameters. 
    void storeIncidence(uint32_t* d_nIncidence, uint32_t h_simRun);                     // Store the dengue incidence after each simulation run.
    double calcLogLikelihood(DengueData* h_dengueData, SimulationParam* h_simParam);    // Calculates the log-likelihood of data given parameters and simulation.
    int nextStep(uint32_t h_MCMCStep, DengueData* h_dengueData,                         // Determine if the candidate parameters are accepted or rejected.
        SimulationParam* h_simParam);                                                   
    void reject(uint32_t h_MCMCStep);                                                   // Rejects the current candidate parameter set.
    
    // Writing MCMC data to file.
    void writePriors(IOParam h_ioParam);                                            // Write the priors of each MCMC parameter to file.
    void writeChain(IOParam h_ioParam, uint32_t h_mcmcStep);                        // Writes the chains of each MCMC parameter.
    void writeCandidates(IOParam h_ioParam, uint32_t h_mcmcStep, int accepted);     // Writes the candidates of each MCMC parameter at each step of the MCMC.

};

#endif