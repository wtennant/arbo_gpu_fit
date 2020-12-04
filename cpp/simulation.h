// Header file containing the declarations for user input functions,
// and the initial simulation run which sets up the initial conditions
// for subsequent runs.

#include "curand_kernel.h"      // CUDA random number generation type.
#include "censustypedef.h"      // Type definitions for census data.
#include "ClimateData.h"        // Climate data parameter class.
#include "Parameter.h"          // Parameter space definition.

// Start of header guard
#ifndef INITIAL_H
#define INITIAL_H

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
                const uint32_t maxTime);

#endif