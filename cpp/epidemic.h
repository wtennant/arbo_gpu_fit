// Header file containing declarations for host functions that invoke CUDA kernels
// to determine the number of disease transmission events between sub-populations,
// as well as initialize the epidemic.

#include "censustypedef.h"      // Type definitions for census data.
#include "host_defines.h"       // Allows definition of host functions.
#include "curand_kernel.h"      // CUDA random number generator type.
#include "Parameter.h"          // Parameter space definition.

// Start of the header guard.
#ifndef EPIDEMIC_H
#define EPIDEMIC_H

// Host function which invokes CUDA kernels determine the number of transmission
// events that should occur to individuals of every subpopulation, and completes
// the transmission.
__host__ void epidemic(age* d_nAge,
                       exposed* d_nExposed,
                       history* d_nHistory,
                       infectStatus* d_nInfectStatus,
                       recovery* d_nRecovery,
                       strain* d_nStrain,
                       age* d_mAge,
                       dead* d_mDead,
                       exposed* d_mExposed,
                       infectStatus* d_mInfectStatus,
                       strain* d_mStrain,
                       curandState_t* d_randStates,
                       uint32_t* d_nTransmission,
                       uint32_t* d_mTransmission,
                       uint32_t* d_nAgeOfInfection,
                       uint32_t* d_nAgeOfInfectionCount,
                       const uint32_t* d_nInfectedSubPopCount,
                       const uint32_t* d_mInfectedSubPopCount,
                       const float* d_sparseNetWeight,
                       const uint32_t* d_sparseNetTo,
                       const uint32_t* d_sparseNetLoc,
                       const uint32_t* d_nSubPopCount,
                       const uint32_t* d_mSubPopCount,
                       const uint32_t* d_nInfectedCount,
                       const uint32_t* d_nSubPopIndex,
                       const uint32_t* d_nSubPopLoc,
                       const uint32_t* d_nSubPopSize,
                       const uint32_t* d_mSubPopIndex,
                       const uint32_t* d_mSubPopLoc,
                       const uint32_t* d_mSubPopSize,
                       const uint32_t* d_nSize,
                       const uint32_t* d_mSize,
                       const uint32_t* d_subPopTotal,
                       const uint32_t h_subPopTotal,
                       const uint32_t t,
                       const uint32_t maxTime,
                       const Parameter h_parameter);

#endif