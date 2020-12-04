// Header file containing the declarations for computing the number of
// transmission events from mosquitoes to humans, and then infecting the
// corresponding humans.

#include "censustypedef.h"      // Type definitions for census data.
#include "host_defines.h"       // Allows definition of host functions.
#include "curand_kernel.h"      // CUDA random number generator type.
#include "Parameter.h"          // Parameter space definition.

// Start of the header guard.
#ifndef NEPIDEMIC_H
#define NEPIDEMIC_H

// Compute the number of transmission events from mosquitoes to humans
// across all sub-populations.
__global__ void nVisitingInfected(curandState_t* d_randStates, 
                                  uint32_t* d_visitInfected,
                                  const uint32_t* d_infectedCount,
                                  const uint32_t* d_nSubPopCount,
                                  const uint32_t* d_mSubPopCount,
                                  const float* d_sparseNetWeight,
                                  const uint32_t* d_sparseNetTo,
                                  const uint32_t* d_sparseNetLoc,
                                  const uint32_t* d_subPopTotal,
                                  const Parameter h_parameter);

// Infect humans given the number of transmission events from mosquitoes to
// humans in each subpopulation for each strain.
__global__ void mnTransmission(age* d_nAge,
                               exposed* d_nExposed,
                               history* d_nHistory,
                               infectStatus* d_nInfectStatus,
                               recovery* d_nRecovery,
                               strain* d_nStrain,
                               curandState_t* d_randStates,
                               uint32_t* d_nAgeOfInfection,
                               uint32_t* d_nAgeOfInfectionCount,
                               const uint32_t* d_nInfectedCount,
                               const uint32_t* d_nSubPopIndex,
                               const uint32_t* d_nSubPopLoc,
                               const uint32_t* d_nSubPopSize,
                               const uint32_t* d_mnTransmission,
                               const uint32_t* d_nSize,
                               const uint32_t* d_subPopTotal,
                               const uint32_t t,
                               const uint32_t maxTime,
                               const Parameter h_parameter);

#endif