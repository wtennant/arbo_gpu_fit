// MCMC.cpp: All member functions of the MCMC class, used in estimating the posterior
// distributions of several pre-compile time determined unobserved parameters.

#include <cstdint>                  // Fixed-width integers.
#include <fstream>                  // Writing to files.
#include <iostream>                 // Input-output stream.
#include "cuda_runtime_api.h"       // CUDA memory copies.
#include "gsl/gsl_linalg.h"         // Cholesky decomposition.
#include "gsl/gsl_randist.h"        // Random number generation distributions.
#include "gsl/gsl_statistics.h"     // Calculating covariance.
#include "ctime"                    // Set the random number generator seed.
#include "IOParam.h"                // Parameter class for input-output files.
#include "SimulationParam.h"        // Simulation parameter class.
#include "DengueData.h"             // Dengue data class.
#include "MCMC.h"                   // Data class for MCMC.
#include "constant.h"               // Constants to the simulation.
#include <gsl/gsl_sf_gamma.h>

// Constructor for the MCMC data class.
MCMC::MCMC(uint32_t input_maxStep, uint32_t input_maxTime, uint32_t input_maxWeek) :
    maxStep(input_maxStep), maxTime(input_maxTime), maxWeek(input_maxWeek)
{
    // Define the size of the covaraiance data structure.
    chainCov = gsl_matrix_alloc(nParams, nParams);

    // Setup the GSL random number generator.
    rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, time(NULL));

    // Declare memory space for the log-likelihood of each point of each simulation.
    particleLogLikelihood = new double[C_MAXSIMRUN]();

    // Declare memory space for weekly incidence of each simulation.
    weekSimIncidence = new uint32_t[maxWeek*C_MAXSIMRUN]();

    // Setup the addresses to the chains, candidate values and priors of the MCMC parameters ( painfully awkwarddd :( ).
    MCMCParam = new MCMCParameterPointer[nParams]{ &scaleEIP, &scaleMLE, &scaleCC, &scaleHumidity, &minMosToHuman, &dispersion, &observationRate};

    // Declare the memory space for each MCMC parameter.
    for (uint32_t i = 0; i < nParams; ++i)
    {
        *(MCMCParam[i].accepted) = new double[maxStep + 1]();
    }

    // Initialization of each MCMC parameter to the mean of the prior.
    for (uint32_t i = 0; i < nParams; ++i)
    {
        (*(MCMCParam[i].accepted))[0] = MCMCParam[i].prior->mean();
    }

    // Initial the first log-likelihood as very small.
    prevAcceptedLogLikelihood = -100000;
}

// Deconstructor for the MCMC data class.
MCMC::~MCMC()
{
    // Free the covariance and mean of the chain.
    gsl_matrix_free(chainCov);

    // Free the GSL random number generator.
    gsl_rng_free(rng);

    // Delete the allocated space for log-likelihoods.
    delete[] particleLogLikelihood;

    // Free the allocated space for the weekly incidence of each simulation.
    delete[] weekSimIncidence;

    // Delete the allocated space for mean time series of infected humans, and each MCMC parameter.
    for (uint32_t i = 0; i < nParams; ++i)
    {
        delete[] *(MCMCParam[i].accepted);
    }
    delete[] MCMCParam;
}

// Copy all parameters necessary for the simulation to the simulation parameter set.
void MCMC::copyParams(SimulationParam* h_simParam, ClimateData* h_climateData)
{
    h_simParam->scaleEIP = scaleEIP.candidate;
    h_simParam->scaleMLE = scaleMLE.candidate;
    h_simParam->scaleCC = scaleCC.candidate;
    h_simParam->scaleHumidity = scaleHumidity.candidate;
    h_simParam->minMosToHuman = minMosToHuman.candidate;
}

// Calculate the co-variance matrix.
void MCMC::calcCov(uint32_t currentStep)
{
    // Calculate the diagonal and lower triangle of the covariance matrix.
    for (uint32_t i = 0; i < nParams; ++i)
    {
        for (uint32_t j = 0; j < nParams; ++j)
        {
            chainCov->data[i*chainCov->tda + j] = gsl_stats_covariance(*(MCMCParam[i].accepted), 1, *(MCMCParam[j].accepted), 1, currentStep == 0 ? 1 : currentStep);
        }
    }

    // Scale the covariance matrix.
    gsl_matrix_scale(chainCov, pow(2.38, 2.0) / static_cast<double>(nParams));

    // Cholesky decomposition of the covariance matrix for use in sampling form the multivariate normal.
    gsl_linalg_cholesky_decomp(chainCov);
}

// Sample candidate from proposal distribution for each parameter.
void MCMC::propose(uint32_t currentStep)
{
    // Define the proposal.
    gsl_vector* proposal = gsl_vector_alloc(nParams);

    // Define the identity matrix.
    gsl_matrix* identity = gsl_matrix_alloc(nParams, nParams);
    gsl_matrix_set_identity(identity);

    // Scale the identity (Roberts 2009).
    gsl_matrix_scale(identity, pow(0.1, 2.0) / static_cast<double>(nParams));
    gsl_linalg_cholesky_decomp(identity);

    // Setup the gsl vector the previously accepted parameter values.
    gsl_vector* prevAccepted = gsl_vector_alloc(nParams);
    for (int i = 0; i < nParams; ++i)
    {
        prevAccepted->data[i] = (*(MCMCParam[i].accepted))[currentStep];
    }

    // Calculate the candidates scaled.
    gsl_ran_multivariate_gaussian(rng, prevAccepted, identity, proposal);

    // Define the small positive constant.
    double beta = 0.05;
    if (currentStep > 100 * nParams)
    {
        // Calculate the covariance of the chains so far.
        calcCov(currentStep);

        // Allocate a gsl_vector for storing the result of sampling from a multivariate normal.
        gsl_vector* result = gsl_vector_alloc(nParams);
        gsl_ran_multivariate_gaussian(rng, prevAccepted, chainCov, result);

        // Add to result.
        gsl_vector_scale(proposal, beta);
        gsl_vector_scale(result, 1.0 - beta);
        gsl_vector_add(proposal, result);

        // Free the result gsl vector.
        gsl_vector_free(result);
    }

    // For each MCMC parameter, sample a new candidate from the proposal distribution.
    for (uint32_t i = 0; i < nParams; ++i)
    {
        *(MCMCParam[i].candidate) = proposal->data[i];
    }
    
    // Free the proposal and previous accepted gsl vectors.
    gsl_vector_free(proposal);
    gsl_vector_free(prevAccepted);
    gsl_matrix_free(identity);
}

// Checks that the candidate parameters make sense (are in the correct range).
bool MCMC::checkParams()
{
    // Set up a boolean to return: 0 for all parameters are valid, 1 for a parameter doesn't make sense.
    bool error = 0;

    // First check that all parameters are positive.
    for (uint32_t i = 0; i < nParams; ++i)
    {
        if (*(MCMCParam[i].candidate) < 0)
        {
            error = 1;
        }
    }

    // Next ensure that some parameters are less than one.
    if ((dispersion.candidate > 1) || (observationRate.candidate > 1))
    {
        error = 1;
    }

    // Return the error handle.
    return error;
}

// Collates the incidence data into an appropriate format to match the dengue data
// and is stored for likelihood calculation.
void MCMC::storeIncidence(uint32_t* d_nIncidence, uint32_t h_simRun)
{
    // Initialize a vector on the host for containing the total number of infected humans
    // per time step for the current simulation.
    uint32_t* h_nIncidence = new uint32_t[C_STRAINS*(maxTime + 1)]();

    // Copy the infected data from the device to the host.
    cudaMemcpy(h_nIncidence, d_nIncidence, sizeof(uint32_t)*C_STRAINS*(maxTime + 1), cudaMemcpyDeviceToHost);

    for (uint32_t w = 0; w < maxWeek; ++w)
    {
        weekSimIncidence[maxWeek*h_simRun + w] = h_nIncidence[7 * w];
        for (uint32_t t = 1; t < 7; ++t)
        {
            if (w * 7 + t <= maxTime)
            {
                weekSimIncidence[maxWeek*h_simRun + w] += h_nIncidence[7 * w + t];
            }
        }
    }

    // Free the memory for human incidence.
    delete[] h_nIncidence;
}

// k successes, r failures, p probability of success.
double negbinomial_pdf(unsigned int k, double r, double p)
{
    double result = gsl_sf_lngamma(k + r) - gsl_sf_lnfact(k) - gsl_sf_lngamma(r);
    result += k * log(p) + r * log(1 - p);
    return result;
}

// The dengue data is often incidence per week, the simulation is incidence per day.
// Calculates the log likelihood of each point in the data given simulated data.
void MCMC::calcEachLogLikelihood(DengueData* h_dengueData, SimulationParam* h_simParam)
{
    // For each simulation calculate the log-likelihood.
    for (uint32_t simRun = 0; simRun < C_MAXSIMRUN; ++simRun)
    {
        // Assess the likelihood (expensive with a binomial distribution).
        double logLikelihood = 0;
        unsigned int cumulativeData;
        double cumulativeIncidence;
        double probOfSuccess;

        // For every week of the year compare with the real data. Ignore the two weeks
        // of incidence data as intialization of the IBM is challenging.
        for (uint32_t w = 1; w < maxWeek; ++w)
        {
            cumulativeData = static_cast<unsigned int>(h_dengueData->incidence[w]);
            cumulativeIncidence = observationRate.candidate*static_cast<double>(weekSimIncidence[maxWeek*simRun + w]);
            probOfSuccess = dispersion.candidate*cumulativeIncidence / (1 + dispersion.candidate*cumulativeIncidence);
            logLikelihood += fmax(-250, negbinomial_pdf(cumulativeData, 1.0 / dispersion.candidate, probOfSuccess));
        }

        // Store the log-likelihood of the time series.
        particleLogLikelihood[simRun] = logLikelihood;
    }
}

// Combines the loglikehood of all time points and simulations.
double MCMC::calcLogLikelihood(DengueData* h_dengueData, SimulationParam* h_simParam)
{
    // Calculate the log-likelihood for each simulation.
    calcEachLogLikelihood(h_dengueData, h_simParam);

    // Calculate the log-likelihood of the current iteration of the MCMC.
    double logLikelihood = 0;

    // Average across each simulation first.
    // Calcualte the maximum value across simulations.
    double maxVal = particleLogLikelihood[0];
    for (int simRun = 1; simRun < C_MAXSIMRUN; ++simRun)
    {
        double c = particleLogLikelihood[simRun];
        if (maxVal < c)
        {
            maxVal = c;
        }
    }

    // Use mathematical wizardy to not cause precision issues when exponentiating and averaging across each time series.
    double logSumExp = 0;
    for (int simRun = 0; simRun < C_MAXSIMRUN; ++simRun)
    {
        logSumExp += exp(particleLogLikelihood[simRun] - maxVal);
    }
    logLikelihood = maxVal + log(logSumExp) - log(C_MAXSIMRUN);

    // Return the log-likelihood
    return logLikelihood;
}

int MCMC::nextStep(uint32_t h_MCMCStep, DengueData* h_dengueData, SimulationParam* h_simParam)
{
    // Calculate the log likelihood of the data given the simulated data.
    logLikelihood = calcLogLikelihood(h_dengueData, h_simParam);

    // Calculcate the loglikelihood of the priors.
    for (uint32_t i = 0; i < nParams; ++i)
    {
        logLikelihood += log(MCMCParam[i].prior->pdf(*(MCMCParam[i].candidate)));
    }

    // Print the log likelihood ratio (provided this isn't the first value in the chain).
    if (h_MCMCStep > 0)
    {
        std::cout << "\nLog-likelihood ratio = " << logLikelihood - prevAcceptedLogLikelihood;
    }

    // Calculate the acceptance ratio.
    double acceptanceRatio = ((h_MCMCStep > 0) ? exp(logLikelihood - prevAcceptedLogLikelihood) : 1);

    // Define variable for if the candidate parameter set is accepted or not.
    int accepted;

    // If accepted, update each parameter and save the current likelihood.
    // Otherwise, keep each parameter the same and use the previous likelihood.
    if (gsl_ran_flat(rng, 0, 1) < acceptanceRatio)
    {
        std::cout << "\n          ";
        for (uint32_t i = 0; i < nParams; ++i)
        {
            std::cout << ((i == 0) ? "" : ", ") << *(MCMCParam[i].name);
        }
        std::cout << std::endl;
        std::cout << "ACCEPTED: ";
        for (uint32_t i = 0; i < nParams; ++i)
        {
            std::cout << ((i == 0)? "" : ", ") << *(MCMCParam[i].candidate);
        }
        std::cout << std::endl;
        
        // Update the chain to include the accepted parameters.
        for (uint32_t i = 0; i < nParams; ++i)
        {
            (*(MCMCParam[i].accepted))[h_MCMCStep + 1] = *(MCMCParam[i].candidate);
        }

        // Update the most recent accepted log-likelihood.
        prevAcceptedLogLikelihood = logLikelihood;

        // Set the acceptance code to one for the candidate parameter set.
        accepted = 1;

        // Update and print out the acceptance rate.
        acceptanceRate = acceptanceRate * (h_MCMCStep / static_cast<double>(h_MCMCStep + 1)) + 1 / static_cast<double>(h_MCMCStep + 1);
        std::cout << "Acceptance rate: " << acceptanceRate * 100 << "%" << std::endl << std::endl;
    }
    else
    {
        // Reject the current candidate parameter set.
        reject(h_MCMCStep);

        // Set the acceptance code to zero for the candidate parameter set.
        accepted = 0;
    }

    // Return the acceptance code for the completed step.
    return accepted;
}

// Rejects the current candidate parameter set.
void MCMC::reject(uint32_t h_MCMCStep)
{
    // Print rejection information to screen.
    std::cout << "\n          ";
    for (uint32_t i = 0; i < nParams; ++i)
    {
        std::cout << ((i == 0) ? "" : ", ") << *(MCMCParam[i].name);
    }
    std::cout << std::endl;
    std::cout << "REJECTED: ";
    for (uint32_t i = 0; i < nParams; ++i)
    {
        std::cout << ((i == 0) ? "" : ", ") << *(MCMCParam[i].candidate);
    }
    std::cout << std::endl;

    // As candidates were rejected, update the chain using previous accepted values.
    for (uint32_t i = 0; i < nParams; ++i)
    {
        (*(MCMCParam[i].accepted))[h_MCMCStep + 1] = (*(MCMCParam[i].accepted))[h_MCMCStep];
    }

    // Update and print out the acceptance rate.
    acceptanceRate = acceptanceRate * (h_MCMCStep / static_cast<double>(h_MCMCStep + 1));
    std::cout << "Acceptance rate: " << acceptanceRate * 100 << "%" << std::endl << std::endl;

}

// Write to file the priors of each MCMC parameter.
void MCMC::writePriors(const IOParam h_ioParam)
{
    // Open the file for writing.
    std::ofstream file_MCMC_priors(h_ioParam.get_odir() + "/MCMC_priors.csv");

    // Write the headers for each MCMC parameter.
    file_MCMC_priors << "PARAMETER_NAME" << "," << "PRIOR_DISTRIBUTION" << "," << "PARAMETERS";
    for (uint32_t i = 0; i < nParams; ++i)
    {
        file_MCMC_priors << "\n" << *(MCMCParam[i].name) << "," << MCMCParam[i].prior->name << "," << MCMCParam[i].prior->printParams();
    }

    // Close the file.
    file_MCMC_priors.close();
}

// Write to file the chains of each MCMC parameter.
void MCMC::writeChain(const IOParam h_ioParam, uint32_t h_mcmcStep)
{
    // Open the file for writing.
    std::ofstream file_MCMC;

    // Create a new file or recording the chains at the start of the MCMC.
    if (h_mcmcStep == 0)
    {
        // Keep trying to open the file if it cannot be opened (being edited elsewhere).
        do
        {
            file_MCMC.open(h_ioParam.get_odir() + "/MCMC.csv", std::ofstream::out);
        } while (!file_MCMC.is_open());

        // Write the headers for each MCMC parameter.
        file_MCMC << "step" << ",logLikelihood";
        for (uint32_t i = 0; i < nParams; ++i)
        {
            file_MCMC << "," << *(MCMCParam[i].name);
        }
    }
    else
    {
        // Append to the chains thereafter.
        do
        {
            file_MCMC.open(h_ioParam.get_odir() + "/MCMC.csv", std::ofstream::out | std::ofstream::app);
        } while (!file_MCMC.is_open());
    }

    // For each MCMC parameter, write the chain for each MCMC parameter.
    int step = h_mcmcStep + 1;
    file_MCMC << "\n" << step - 1 << "," << prevAcceptedLogLikelihood;
    for (uint32_t i = 0; i < nParams; ++i)
    {
        file_MCMC << "," << (*(MCMCParam[i].accepted))[step];
    }

    // Close the file.
    file_MCMC.close();
}

// Write to file the candidates of each MCMC parameter at each step of the MCMC.
void MCMC::writeCandidates(const IOParam h_ioParam, uint32_t h_mcmcStep, int accepted)
{
    // Open the file for writing.
    std::ofstream file_MCMC_candidates;

    // Create a new file or recording the candidates at the start of the MCMC.
    if (h_mcmcStep == 0)
    {
        // Keep trying to open the file if it cannot be opened (being edited elsewhere).
        do
        {
            file_MCMC_candidates.open(h_ioParam.get_odir() + "/MCMC_candidates.csv", std::ofstream::out);
        } while (!file_MCMC_candidates.is_open());

        // Write the headers for each MCMC parameter.
        file_MCMC_candidates << "step" << ",logLikelihood";
        for (uint32_t i = 0; i < nParams; ++i)
        {
            file_MCMC_candidates << "," << *(MCMCParam[i].name);
        }
        file_MCMC_candidates << ",accepted";
    }
    else
    {
        // Append to the candidates thereafter.
        do
        {
            file_MCMC_candidates.open(h_ioParam.get_odir() + "/MCMC_candidates.csv", std::ofstream::out | std::ofstream::app);
        } while (!file_MCMC_candidates.is_open());
    }

    // For each MCMC parameter, write the candidate values for each MCMC parameter.
    file_MCMC_candidates << "\n" << h_mcmcStep << "," << logLikelihood;
    for (uint32_t i = 0; i < nParams; ++i)
    {
        file_MCMC_candidates << "," << *(MCMCParam[i].candidate);
    } 
    file_MCMC_candidates << "," << accepted;

    // Close the file.
    file_MCMC_candidates.close();
}