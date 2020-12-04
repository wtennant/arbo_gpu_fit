// main.cpp: the root file for the individual-based stochastic dengue model.

#include <iostream>				// Input/output to console.
#include <windows.h>            // Create output directory.
#include <random>				// (Pseudo-)random number generation.
#include <chrono>				// Seed for RNG.
#include "censustypedef.h"      // Type definitions for census data.
#include "network.h"            // Network creation and analysis.
#include "simulation.h"         // Dengue simulation.
#include "Parameter.h"          // Parameter class.
#include "Data.h"               // Data class.
#include "MCMC.h"               // MCMC class.
#include "constant.h"			// Constants for simulation.

// Root function for initialization of simulation variables, and function
// calling to data collecting, demographic simulations and epidemiological simulations.
int main(int argc, char* argv[])
{
    // Initialize the class of all parameters.
    Parameter h_parameter;

	// Parse the input command arguments into input and output folder.
	h_parameter.io.commandLineParse(argc, argv);

    // Create the input and output file directory if necessary.
    CreateDirectoryA(h_parameter.io.get_odir().c_str(), NULL);
    CreateDirectoryA(h_parameter.io.get_idir().c_str(), NULL);

    // Initialize the class containing data, including climate and dengue incidence data.
    Data h_data(h_parameter.io);

    // The number of days to simulate is one less than the total number of days
    // in the climate data file.
    uint32_t maxTime = h_data.dengue.maxWeek*7 - 1;

    // initialize the class containing all MCMC related data, including
    // the maximum number of MCMC chains, and most recent likelihood calculation.
    MCMC h_MCMC(C_MAXMCMCSTEPS, maxTime, h_data.dengue.maxWeek);

    // Write the priors for each parameter to file.
    h_MCMC.writePriors(h_parameter.io);

    // If a lattice is requested, adjust total number of sub-populations
    // to form a square.
    h_parameter.simulation.squareLattice();

    // Initialize the total number of subpopulations.
    uint32_t h_subPopTotal = static_cast<uint32_t>(h_parameter.simulation.metaPop);

    // Declare and initialize the network (does not change in MCMC).
    // Get the number of connections in the graph.
    igraph_matrix_t h_network;
    uint32_t h_netEdges;
    h_netEdges = createNetwork(&h_network, h_parameter, h_subPopTotal);

    // Declare arrays for the non-zero weights in the network,
    // the locations in the weight vector of each new row of the adjacency matrix
    // and the column indices associated with the non-zero weights.
    float* h_sparseNetWeight = new float[2 * h_netEdges + h_subPopTotal];
    uint32_t* h_sparseNetTo = new uint32_t[2 * h_netEdges + h_subPopTotal];
    uint32_t* h_sparseNetLoc = new uint32_t[h_subPopTotal + 1];

    // Convert the igraph sparse matrix to the compressed sparse row format.
    igraphSparseToCSR(h_sparseNetWeight, h_sparseNetTo, h_sparseNetLoc, h_network, h_subPopTotal);

    // Destroy the igraph adjacency matrix.
    igraph_matrix_destroy(&h_network);

    // For each step of the MCMC chain, move to the next step of the MCMC, and
    // perform the simulation a fixed number of times, to generate the liklihood of the
    // parameters given the data.
    for (uint32_t h_mcmcStep = 0; h_mcmcStep < C_MAXMCMCSTEPS; ++h_mcmcStep)
    {
        // Print the current step of the MCMC to console.
        std::cout << "MCMC step: " << h_mcmcStep << std::endl;

        // Sample new candidates parameters.
        h_MCMC.propose(h_mcmcStep);
        
        // Define a variable for if the candidate parameter set is accepted or rejected.
        int accepted;

        // Check if the proposed parameters are valid.
        if (h_MCMC.checkParams())
        {
            std::cout << "One or more candidate parameters are not valid";

            // Reject the candidate parameters.
            h_MCMC.reject(h_mcmcStep);
            accepted = 0;
        }
        else
        {
            // Change simulation parameters from MCMC.
            h_MCMC.copyParams(&h_parameter.simulation, &h_data.climate);

            // Save to file the important constants/parameters for the simulation run.
            h_parameter.writeParameters(h_mcmcStep);

            // Declare arrays for the normalized cumulative community size for humans and mosquitoes.
            float* h_nCumulativeComSize = new float[h_subPopTotal];
            float* h_mCumulativeComSize = new float[h_subPopTotal];

            // Compute normalized cumulative community size.
            calcCumulativeComSize(h_nCumulativeComSize, h_mCumulativeComSize, h_sparseNetLoc, h_parameter.simulation, h_subPopTotal);

            // Read in the initial human population size, and the dimensions of the metapopulation lattice
            // from the parameter data.
            uint32_t h_nSize = static_cast<uint32_t>(h_parameter.simulation.nSize);

            // Initialize the maximum mosquito population size. This is the size at time zero of the simulation.
            uint32_t h_mSize = static_cast<uint32_t>(h_parameter.simulation.minMosToHuman * pow(2, h_parameter.simulation.scaleCC) * h_nSize);

            // Initialize the total number of infected individuals at the start of the simulation.
            uint32_t nInitialInfected = h_parameter.simulation.nInitialInf;
            uint32_t mInitialInfected = h_parameter.simulation.mInitialInf;

            // Declare the GPU device variables for human census data.
            age* d_nAge;                        // Individual age (days)
            exposed* d_nExposed;                // Age at which infection becomes infectious.
            history* d_nHistory;                // Strains an individual is immune to.
            infectStatus* d_nInfectStatus;      // If the individual is susceptible, infected, or infectious.
            pLifeExpectancy* d_nPLE;            // Random probability determing life expectancy of the individual.
            recovery* d_nRecovery;              // Age at which infection ends.
            strain* d_nStrain;                  // Dengue serotype an individual is infected with.
            subPopulation* d_nSubPopulation;    // Subpopulation that the individual belongs to.

            // Declare the GPU device variables for the mosquito census data.
            age* d_mAge;                        // Individual age (days)
            dead* d_mDead;                      // Alive or dead
            exposed* d_mExposed;                // Age at which infection becomes infectious.
            infectStatus* d_mInfectStatus;      // If the individual is susceptible, infected, or infectious.
            pLifeExpectancy* d_mPLE;            // Random probability determing life expectancy of the individual.
            strain* d_mStrain;                  // Dengue serotype an individual is infected with.
            subPopulation* d_mSubPopulation;    // Subpopulation that the individual belongs to.

            // Declare the device variables used in the initialization and demographic update.
            float* d_nSurvival;                 // The human cumulative survival function.
            float* d_mSurvival;                 // The mosquito cumulative survival function.
            float* d_mExpectedPopSize;          // Expected population size of mosquitoes.
            float* d_nCumulativeComSize;        // The normalized cumulative size of each human community.
            float* d_mCumulativeComSize;        // The normalized cumulative size of each mosquito community.

            // Declare the GPU device variables for counting different sets of individuals.
            uint16_t *d_mDeadCount;                 // Number of dead mosquitoes per GPU block.
            uint32_t *d_nSubPopCount;               // Number of humans per subpopulation.
            uint32_t *d_mSubPopCount;               // Number of mosquitoes per subpopulation.
            uint32_t *d_nInfectedSubPopCount;       // Number of infected humans per subpopulation per strain.
            uint32_t *d_mInfectedSubPopCount;       // Number of infected mosquitoes per subpopulation per strain.
            uint32_t *d_nInfectedSubPopCountSeries; // Number of infected humans per subpopulation per strain per day.
            uint32_t *d_nIncidenceSubPop;			// Number of new infections per subpopulation per strain per day.
            uint32_t *d_nInfectedCount;             // Time series for the number of infected humans per strain.
            uint32_t* d_nOneSubPopInfectedCount;    // Time series for the number of infected humans per strain for a specific subpopulation.
            uint32_t *d_nIncidence;					// Time series for the number of new infections per strain.
            uint32_t *d_nCount;                     // Time series for the number of humans.
            uint32_t *d_mCount;                     // Time series for the number of mosquitoes.
            uint32_t *d_nReductionInfectedCount;    // Used in summing the number of infected humans per strain across all subpopulations.
            uint32_t *d_nReductionCount;            // Used in summing the number of humans across all subpopulations.
            uint32_t *d_mReductionCount;            // Used in summing the number of mosquitoes across all subpopulations.

            // Declare the GPU device variables for disease transmission.
            uint32_t *d_nSubPopIndex, *d_mSubPopIndex;              // Census indices ordered by sub-population.
            uint32_t *d_nSubPopLoc, *d_mSubPopLoc;                  // Indices of the above where a new sub-population begins in the ordering.
            uint32_t *d_nSubPopSize, *d_mSubPopSize;                // The maximum number of individuals per sub-population.
            uint32_t *d_nTransmission, *d_mTransmission;            // Transmission numbers per subpopulation per strain.
            uint32_t *d_nAgeOfInfection, *d_nAgeOfInfectionCount;   // Ages of the last few infections for each novel exposure.
            float *d_sparseNetWeight;					            // The non-zero weights/distances between all communities.
            uint32_t *d_sparseNetLoc;					            // The location in the sparse network weight array where each communities connections start.
            uint32_t *d_sparseNetTo;					            // The communities that each community is connected to.

            // Declare the GPU device variables which are constant after user input.
            uint32_t *d_nSize;          // Human population size.
            uint32_t *d_mSize;          // Maximum mosquito population size.
            uint32_t *d_subPopTotal;    // Total number of subpopulations.

            // Declare the GPU device variables for random number generation on the GPU.
            curandState_t *d_randStates;

            // Initialize the number of blocks required on the GPU given the number of threads desired to be used on
            // each block in order to have a thread per individual.
            uint32_t nGridSize = static_cast<uint32_t>(ceil(h_nSize / static_cast<float>(C_THREADSPERBLOCK)));
            uint32_t mGridSize = static_cast<uint32_t>(ceil(h_mSize / static_cast<float>(C_THREADSPERBLOCK)));

            // Initialize the number of blocks such that each thread on the block is assigned to one subpopulation. This is used
            // in summing count data across all subpopulations.
            uint32_t reductionSize{ static_cast<uint32_t>(ceil(h_subPopTotal / static_cast<float>(C_THREADSPERBLOCKSUM))) };

            // Initialize the total number of active threads on the device at any one time. This will be used for
            // determining the number of random number generators to be created on the device.
            uint32_t totalActiveThreads{ h_parameter.arch.totalSM*h_parameter.arch.warpsPerSM*h_parameter.arch.threadsPerWarp };

            // Compute the maximum time between the user input and the pre-defined initial simulation length.
            // This is to ensure overflow does not occur when recording time series data in the initial simulation.
            uint32_t timeSeriesMaxTime = static_cast<uint32_t>(maxTime);

            // Allocate space for the device variables onto the device. Allocate (roughly)
            // from the largest in size to the smallest.
            cudaMalloc((void **)&d_nAge, sizeof(age)*h_nSize);
            cudaMalloc((void **)&d_nExposed, sizeof(exposed)*h_nSize);
            cudaMalloc((void **)&d_nHistory, sizeof(history)*h_nSize*C_STRAINS);
            cudaMalloc((void **)&d_nInfectStatus, sizeof(infectStatus)*h_nSize);
            cudaMalloc((void **)&d_nPLE, sizeof(pLifeExpectancy)*h_nSize);
            cudaMalloc((void **)&d_nRecovery, sizeof(recovery)*h_nSize);
            cudaMalloc((void **)&d_nStrain, sizeof(strain)*h_nSize);
            cudaMalloc((void **)&d_nSubPopulation, sizeof(subPopulation)*h_nSize);
            cudaMalloc((void **)&d_mAge, sizeof(age)*h_mSize);
            cudaMalloc((void **)&d_mDead, sizeof(dead)*h_mSize);
            cudaMalloc((void **)&d_mExposed, sizeof(exposed)*h_mSize);
            cudaMalloc((void **)&d_mInfectStatus, sizeof(infectStatus)*h_mSize);
            cudaMalloc((void **)&d_mPLE, sizeof(pLifeExpectancy)*h_mSize);
            cudaMalloc((void **)&d_mStrain, sizeof(strain)*h_mSize);
            cudaMalloc((void **)&d_mSubPopulation, sizeof(subPopulation)*h_mSize);
            cudaMalloc((void **)&d_nSubPopIndex, sizeof(uint32_t)*h_nSize);
            cudaMalloc((void **)&d_mSubPopIndex, sizeof(uint32_t)*h_mSize);
            cudaMalloc((void **)&d_randStates, sizeof(curandState_t)*totalActiveThreads);
            cudaMalloc((void **)&d_nInfectedSubPopCountSeries, sizeof(uint32_t)*h_subPopTotal*C_STRAINS*(timeSeriesMaxTime + 1));
            cudaMalloc((void **)&d_nInfectedSubPopCount, sizeof(uint32_t)*h_subPopTotal*C_STRAINS);
            cudaMalloc((void **)&d_mInfectedSubPopCount, sizeof(uint32_t)*h_subPopTotal*C_STRAINS);
            cudaMalloc((void **)&d_nIncidenceSubPop, sizeof(uint32_t)*h_subPopTotal*C_STRAINS);
            cudaMalloc((void **)&d_nTransmission, sizeof(uint32_t)*h_subPopTotal*C_STRAINS);
            cudaMalloc((void **)&d_mTransmission, sizeof(uint32_t)*h_subPopTotal*C_STRAINS);
            cudaMalloc((void **)&d_nAgeOfInfection, sizeof(uint32_t)*C_NAOIRECORD*C_STRAINS);
            cudaMalloc((void **)&d_sparseNetWeight, sizeof(float)*h_netEdges * 2);
            cudaMalloc((void **)&d_sparseNetTo, sizeof(uint32_t)*h_netEdges * 2);
            cudaMalloc((void **)&d_sparseNetLoc, sizeof(uint32_t)*(h_subPopTotal + 1));
            cudaMalloc((void **)&d_nCumulativeComSize, sizeof(float)*h_subPopTotal);
            cudaMalloc((void **)&d_mCumulativeComSize, sizeof(float)*h_subPopTotal);
            cudaMalloc((void **)&d_nSubPopCount, sizeof(uint32_t)*h_subPopTotal);
            cudaMalloc((void **)&d_mSubPopCount, sizeof(uint32_t)*h_subPopTotal);
            cudaMalloc((void **)&d_nSubPopLoc, sizeof(uint32_t)*h_subPopTotal);
            cudaMalloc((void **)&d_mSubPopLoc, sizeof(uint32_t)*h_subPopTotal);
            cudaMalloc((void **)&d_nSubPopSize, sizeof(uint32_t)*h_subPopTotal);
            cudaMalloc((void **)&d_mSubPopSize, sizeof(uint32_t)*h_subPopTotal);
            cudaMalloc((void **)&d_mDeadCount, sizeof(uint16_t)*mGridSize);
            cudaMalloc((void **)&d_nSurvival, sizeof(float)*(C_NMAXINITIALAGE + 1));
            cudaMalloc((void **)&d_mSurvival, sizeof(float)*(C_MMAXINITIALAGE + 1));
            cudaMalloc((void **)&d_nReductionInfectedCount, sizeof(uint32_t)*C_STRAINS*reductionSize);
            cudaMalloc((void **)&d_nReductionCount, sizeof(uint32_t)*reductionSize);
            cudaMalloc((void **)&d_mReductionCount, sizeof(uint32_t)*reductionSize);
            cudaMalloc((void **)&d_nOneSubPopInfectedCount, sizeof(uint32_t)*C_STRAINS*(timeSeriesMaxTime + 1));
            cudaMalloc((void **)&d_nInfectedCount, sizeof(uint32_t)*C_STRAINS*(timeSeriesMaxTime + 1));
            cudaMalloc((void **)&d_nIncidence, sizeof(uint32_t)*C_STRAINS*(timeSeriesMaxTime + 1));
            cudaMalloc((void **)&d_nCount, sizeof(uint32_t)*(timeSeriesMaxTime + 1));
            cudaMalloc((void **)&d_mCount, sizeof(uint32_t)*(timeSeriesMaxTime + 1));
            cudaMalloc((void **)&d_nAgeOfInfectionCount, sizeof(uint32_t)*C_STRAINS);
            cudaMalloc((void **)&d_nSize, sizeof(uint32_t));
            cudaMalloc((void **)&d_mSize, sizeof(uint32_t));
            cudaMalloc((void **)&d_subPopTotal, sizeof(uint32_t));
            cudaMalloc((void **)&d_mExpectedPopSize, sizeof(float));

            // Copy memory from the host to the allocated space in the device.
            cudaMemcpy(d_sparseNetWeight, h_sparseNetWeight, sizeof(float)*h_netEdges * 2, cudaMemcpyHostToDevice);
            cudaMemcpy(d_sparseNetTo, h_sparseNetTo, sizeof(uint32_t)*h_netEdges * 2, cudaMemcpyHostToDevice);
            cudaMemcpy(d_sparseNetLoc, h_sparseNetLoc, sizeof(uint32_t)*(h_subPopTotal + 1), cudaMemcpyHostToDevice);
            cudaMemcpy(d_nCumulativeComSize, h_nCumulativeComSize, sizeof(float)*h_subPopTotal, cudaMemcpyHostToDevice);
            cudaMemcpy(d_mCumulativeComSize, h_mCumulativeComSize, sizeof(float)*h_subPopTotal, cudaMemcpyHostToDevice);
            cudaMemcpy(d_subPopTotal, &h_subPopTotal, sizeof(uint32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_nSize, &h_nSize, sizeof(uint32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_mSize, &h_mSize, sizeof(uint32_t), cudaMemcpyHostToDevice);

            // Run the simulation on a pre-compile-time defined number of parameter sets.
            for (uint32_t h_simRun = 0; h_simRun < C_MAXSIMRUN; ++h_simRun)
            {
                // Output the simulation number.
                std::cout << "\rSimulation " << h_simRun + 1 << " of " << C_MAXSIMRUN;

                // Run the dengue simulation for a long period of time in order to setup
                // initial conditions for future runs of the simulation. The initial simulation
                // also helps with the post-program calculation of R0.
                simulation(d_nAge, d_nExposed, d_nHistory, d_nInfectStatus, d_nPLE, d_nRecovery, d_nStrain, d_nSubPopulation,
                    d_mAge, d_mDead, d_mExposed, d_mInfectStatus, d_mPLE, d_mStrain, d_mSubPopulation, d_randStates, d_nSubPopIndex, d_mSubPopIndex,
                    d_nInfectedSubPopCount, d_mInfectedSubPopCount, d_nIncidenceSubPop, d_nTransmission, d_mTransmission, d_nAgeOfInfection, d_nAgeOfInfectionCount,
                    d_nSubPopCount, d_mSubPopCount, d_nSubPopLoc, d_mSubPopLoc, d_nSubPopSize, d_mSubPopSize, d_mDeadCount, d_nSurvival, d_mSurvival,
                    d_nReductionInfectedCount, d_nReductionCount, d_mReductionCount, d_nInfectedSubPopCountSeries, d_nOneSubPopInfectedCount, d_nInfectedCount, d_nIncidence,
                    d_nCount, d_mCount, d_nSize, d_mSize, d_subPopTotal, h_parameter, d_mExpectedPopSize, d_sparseNetWeight, d_sparseNetTo, d_sparseNetLoc, d_nCumulativeComSize,
                    d_mCumulativeComSize, nInitialInfected, mInitialInfected, nGridSize, mGridSize, h_nSize, h_mSize, h_subPopTotal, h_data.climate, h_simRun, h_mcmcStep, maxTime);

                // After each simulation, calculate the log-likelihood of each time point in the simulation.
                h_MCMC.storeIncidence(d_nIncidence, h_simRun);
            }

            // Free the allocated space on the device for the device variables (this will prevent memory leaks). Furthermore,
            // it was suggested to adopt a "Last In, First Out" (LIFO) strategy.
            cudaFree(d_mExpectedPopSize);
            cudaFree(d_subPopTotal);
            cudaFree(d_mSize);
            cudaFree(d_nSize);
            cudaFree(d_mCount);
            cudaFree(d_nCount);
            cudaFree(d_nIncidence);
            cudaFree(d_nInfectedCount);
            cudaFree(d_nOneSubPopInfectedCount);
            cudaFree(d_mReductionCount);
            cudaFree(d_nReductionCount);
            cudaFree(d_nReductionInfectedCount);
            cudaFree(d_nAgeOfInfectionCount);
            cudaFree(d_mSurvival);
            cudaFree(d_nSurvival);
            cudaFree(d_mDeadCount);
            cudaFree(d_mSubPopSize);
            cudaFree(d_nSubPopSize);
            cudaFree(d_mSubPopLoc);
            cudaFree(d_nSubPopLoc);
            cudaFree(d_mSubPopCount);
            cudaFree(d_nSubPopCount);
            cudaFree(d_nCumulativeComSize);
            cudaFree(d_mCumulativeComSize);
            cudaFree(d_sparseNetLoc);
            cudaFree(d_sparseNetTo);
            cudaFree(d_sparseNetWeight);
            cudaFree(d_nAgeOfInfection);
            cudaFree(d_mTransmission);
            cudaFree(d_nTransmission);
            cudaFree(d_nIncidenceSubPop);
            cudaFree(d_mInfectedSubPopCount);
            cudaFree(d_nInfectedSubPopCount);
            cudaFree(d_nInfectedSubPopCountSeries);
            cudaFree(d_randStates);
            cudaFree(d_mSubPopIndex);
            cudaFree(d_nSubPopIndex);
            cudaFree(d_mSubPopulation);
            cudaFree(d_mStrain);
            cudaFree(d_mPLE);
            cudaFree(d_mInfectStatus);
            cudaFree(d_mExposed);
            cudaFree(d_mDead);
            cudaFree(d_mAge);
            cudaFree(d_nSubPopulation);
            cudaFree(d_nStrain);
            cudaFree(d_nRecovery);
            cudaFree(d_nPLE);
            cudaFree(d_nInfectStatus);
            cudaFree(d_nHistory);
            cudaFree(d_nExposed);
            cudaFree(d_nAge);

            // Delete all dynamic array space declared on the host.
            delete[] h_nCumulativeComSize;
            delete[] h_mCumulativeComSize;

            // Carry out the MCMC.
            accepted = h_MCMC.nextStep(h_mcmcStep, &h_data.dengue, &h_parameter.simulation);
        }

        // Write the candidate set of MCMC parameters.
        h_MCMC.writeCandidates(h_parameter.io, h_mcmcStep, accepted);

        // Write the MCMC posteriors.
        h_MCMC.writeChain(h_parameter.io, h_mcmcStep);
    }

    // Delete the space allocated for the network.
    delete[] h_sparseNetTo;
    delete[] h_sparseNetLoc;
    delete[] h_sparseNetWeight;

    // Explicitly destroy and clean up all resources associated with the current device in the current process.
    cudaDeviceReset();

    // To overwrite the recording progress bar with "Press any key..."
    std::cout << "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b";

    // Shutdown the machine if requested.
    if (C_SHUTDOWN)
    {
        system("shutdown -s");
    }

    // Standard (no error) return with main().
    return 0;
}