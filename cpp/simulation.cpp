// simulation.cpp: function for running the individual based model for dengue.

#include <iostream>				// Input/output to console.
#include <ctime>                // Simulation timings.
#include "censustypedef.h"      // Type definitions for census data.
#include "setuprng.h"			// CUDA random number intialization.
#include "demographicinitial.h" // Demography initialization.
#include "epidemicinitial.h"    // Epidemic initialization.
#include "demographic.h"		// Human and mosquito demographic updating.
#include "epidemic.h"           // Simulation of transmission events.
#include "datacollect.h"        // Data collecting and collating.
#include "writeoutput.h"        // Writing simulation output to file.
#include "constant.h"			// Constants for the simulation.
#include "Parameter.h"          // Parameter class.

// Runs the dengue simulation for a long time in order to setup
// initial conditions for future runs of the simulation.
void simulation(age* d_nAge,
                exposed* d_nExposed,
                history* d_nHistory,
                infectStatus* d_nInfectStatus,
                pLifeExpectancy* d_nPLE,
                recovery* d_nRecovery,
                strain* d_nStrain,
                subPopulation* d_nSubPopulation,
                age* d_mAge,
                dead* d_mDead,
                exposed* d_mExposed,
                infectStatus* d_mInfectStatus,
                pLifeExpectancy* d_mPLE,
                strain* d_mStrain,
                subPopulation* d_mSubPopulation,
                curandState_t* d_randStates,
                uint32_t* d_nSubPopIndex,
                uint32_t* d_mSubPopIndex,
                uint32_t* d_nInfectedSubPopCount,
                uint32_t* d_mInfectedSubPopCount,
			    uint32_t* d_nIncidenceSubPop,
                uint32_t* d_nTransmission,
                uint32_t* d_mTransmission,
                uint32_t* d_nAgeOfInfection,
                uint32_t* d_nAgeOfInfectionCount,
                uint32_t* d_nSubPopCount,
                uint32_t* d_mSubPopCount,
                uint32_t* d_nSubPopLoc,
                uint32_t* d_mSubPopLoc,
                uint32_t* d_nSubPopSize,
                uint32_t* d_mSubPopSize,
                uint16_t* d_mDeadCount,
                float* d_nSurvival,
                float* d_mSurvival,
                uint32_t* d_nReductionInfectedCount,
                uint32_t* d_nReductionCount,
                uint32_t* d_mReductionCount,
                uint32_t* d_nInfectedSubPopCountSeries,
                uint32_t* d_nOneSubPopInfectedCount,
                uint32_t* d_nInfectedCount,
			    uint32_t* d_nIncidence,
                uint32_t* d_nCount,
                uint32_t* d_mCount,
                uint32_t* d_nSize,
                uint32_t* d_mSize,
                uint32_t* d_subPopTotal,
                Parameter h_parameter,
                float* d_mExpectedPopSize,
                const float* d_sparseNetWeight,
                const uint32_t* d_sparseNetTo,
                const uint32_t* d_sparseNetLoc,
                const float* d_nCumulativeComSize,
                const float* d_mCumulativeComSize,
                const uint32_t nInitialInfected,
                const uint32_t mInitialInfected,
                const uint32_t nGridSize,
                const uint32_t mGridSize,
                const uint32_t h_nSize,
                const uint32_t h_mSize,
                const uint32_t h_subPopTotal,
                const ClimateData h_climateData,
                const uint32_t h_simRun,
                const uint32_t h_mcmcRun,
                const uint32_t maxTime)
{
    // Initialize the array containing the expected size of the mosquito population size at each time step.
    float* h_mExpectedPopSize = new float[maxTime + 1];

    // Initialize the mosquito population size.
    h_mExpectedPopSize[0] = h_parameter.simulation.minMosToHuman*pow(1 + h_climateData.precipitation[0], h_parameter.simulation.scaleCC)*h_nSize;
    cudaMemcpy(d_mExpectedPopSize, &h_mExpectedPopSize[0], sizeof(float), cudaMemcpyHostToDevice);

    // Set up the random number generators on the GPU.
    setupCudaRNG(d_randStates, h_parameter.arch);

    // Initialize the start time for the simulation.
    std::clock_t start{ std::clock() };
    float duration;

    // Adjust epidemilogical parameters depending on the climate at the initial time step.
    h_parameter.simulation.adjustClimate(h_climateData, 0);

    // Initialize the human and mosquito populations.
    demographicInitialization(d_nAge, d_nHistory, d_nInfectStatus, d_nPLE, d_nSubPopulation, d_mAge, d_mDead, d_mInfectStatus, d_mPLE, d_mSubPopulation,
        d_nSubPopIndex, d_mSubPopIndex, d_randStates, d_nSubPopCount, d_mSubPopCount, d_nSubPopLoc, d_mSubPopLoc, d_nSubPopSize, d_mSubPopSize,
        d_mDeadCount, d_nSurvival, d_mSurvival, d_nCumulativeComSize, d_mCumulativeComSize, d_mExpectedPopSize,
        d_nSize, d_mSize, d_subPopTotal, h_nSize, h_mSize, h_subPopTotal, h_parameter);

    // Initialize the infections in the human and mosquito populations.
    epidemicInitial(d_nAge, d_nHistory, d_nInfectStatus, d_nRecovery, d_nStrain, d_nSubPopulation, d_mInfectStatus, d_mStrain, d_mSubPopulation,
        d_randStates, d_nInfectedSubPopCount, d_mInfectedSubPopCount, d_nIncidenceSubPop, d_nSize, d_subPopTotal,
        nInitialInfected, mInitialInfected, nGridSize, mGridSize, h_subPopTotal, h_parameter);

    // Declare the timestep counter t.
    uint32_t t;

    // For every time step, run the simulation.
    for (t = 0; t < maxTime; ++t)
    {
        // Sum up the number of infected/total individuals across all sub-populations and store the results in time-series.
        dataCollect(d_nReductionInfectedCount, d_nReductionCount, d_mReductionCount, d_nInfectedSubPopCountSeries,
			d_nOneSubPopInfectedCount, d_nInfectedCount, d_nIncidence, d_nCount, d_mCount, d_nInfectedSubPopCount,
			d_nIncidenceSubPop, d_nSubPopCount, d_mSubPopCount, 0, h_subPopTotal, t);

        // Adjust epidemilogical parameters depending on the climate at the next time step.
        h_parameter.simulation.adjustClimate(h_climateData, t + 1);

        // Determine the expected mosquito population size. This is to model the rainfall seasonality of the
        // moqsuito population density.
        // Use vector to human biting rate as a proxy for the seasonal signature of the mosquito carrying
        // capacity.
        unsigned int rainDataTime = t + 1;
        h_mExpectedPopSize[t + 1] = h_mExpectedPopSize[t] + h_parameter.simulation.mosNetGrowth*h_mExpectedPopSize[t] *
            (1.0 - h_mExpectedPopSize[t] / (h_parameter.simulation.minMosToHuman*h_nSize*pow(1.0f + h_climateData.precipitation[rainDataTime], h_parameter.simulation.scaleCC)));
        cudaMemcpy(d_mExpectedPopSize, &h_mExpectedPopSize[t + 1], sizeof(float), cudaMemcpyHostToDevice);

        // Run the human demographics on CUDA. For every individual, the human demographic function
        // will determine if an individual is due to die or not (from natural causes). If not, age
        // the individual by one day.
        nDemographic(d_nAge, d_nExposed, d_nHistory, d_nInfectStatus, d_nPLE, d_nRecovery, d_nStrain, d_nSubPopulation,
            d_randStates, d_nInfectedSubPopCount, d_nIncidenceSubPop, d_nSize, d_subPopTotal, h_nSize, h_parameter);

        // Run the mosquito demographics on CUDA. For every individual, the mosquito demographic function
        // will determine if an individual is due to die or not (from natural causes). If not, age
        // the individual by one day.
        mDemographic(d_mAge, d_mDead, d_mExposed, d_mInfectStatus, d_mPLE, d_mStrain, d_mSubPopulation,
            d_randStates, d_mInfectedSubPopCount, d_mSubPopCount, d_mDeadCount,
            d_mSize, d_subPopTotal, h_mSize, d_mExpectedPopSize, h_parameter);

        // Determine the number of infections that occur from infected individuals, generate that many random numbers to determine
        // which subpopulation the transmission events occur in, and which individuals in those subpopulations they infect.
        epidemic(d_nAge, d_nExposed, d_nHistory, d_nInfectStatus, d_nRecovery, d_nStrain, d_mAge, d_mDead,
            d_mExposed, d_mInfectStatus, d_mStrain, d_randStates, d_nTransmission, d_mTransmission, d_nAgeOfInfection, d_nAgeOfInfectionCount,
            d_nInfectedSubPopCount, d_mInfectedSubPopCount, d_sparseNetWeight, d_sparseNetTo, d_sparseNetLoc,
            d_nSubPopCount, d_mSubPopCount, d_nInfectedCount, d_nSubPopIndex, d_nSubPopLoc, d_nSubPopSize,
            d_mSubPopIndex, d_mSubPopLoc, d_mSubPopSize, d_nSize, d_mSize, d_subPopTotal,
            h_subPopTotal, t, maxTime, h_parameter);
    }

    // Sum up the number of infected/total individuals across all sub-populations one final time.
    dataCollect(d_nReductionInfectedCount, d_nReductionCount, d_mReductionCount, d_nInfectedSubPopCountSeries,
		d_nOneSubPopInfectedCount, d_nInfectedCount, d_nIncidence, d_nCount, d_mCount, d_nInfectedSubPopCount,
		d_nIncidenceSubPop, d_nSubPopCount, d_mSubPopCount, 0, h_subPopTotal, maxTime);

    // Wait for device work to finish.
    cudaDeviceSynchronize();

    // Write the incidence time series to file.
    writeIncidence(d_nIncidence, d_nInfectedCount, d_nOneSubPopInfectedCount, d_nCount, h_parameter.io, h_simRun, h_mcmcRun, maxTime);

    // Writes a partial human census to file.
    if (C_WRITE_CENSUS == 1)
    {
        writeCensus(d_nAge, d_nHistory, h_parameter.io, h_nSize, h_simRun, h_mcmcRun);
    }

    // Writes the ages of the most recent novel infections to file.
    if (C_WRITE_AOI == 1)
    {
        writeAOI(d_nAgeOfInfection, h_simRun, h_mcmcRun, h_parameter.io);
    }

    // Write spatial data to file.
    if (C_WRITE_SPATIAL == 1)
    {
        writeSpatial(d_nInfectedSubPopCountSeries, h_subPopTotal, maxTime, h_simRun, h_mcmcRun, h_parameter.io);
    }    

    // Output the total time for the simulation to run.
    duration = (std::clock() - start) / (float)CLOCKS_PER_SEC;

    // Delete the dynamic-allocated arrays.
    delete[] h_mExpectedPopSize;
}