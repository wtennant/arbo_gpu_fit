// Header file containing the declarations for computing the number of
// transmission events from humans to mosquitoes, and then infecting the
// corresponding mosquitoes.

#include "censustypedef.h"      // Type definitions for census data.
#include "host_defines.h"       // Allows definition of host functions.
#include "curand_kernel.h"      // CUDA random number generator type.
#include "Parameter.h"          // Parameter space definition.

// Start of the header guard.
#ifndef MEPIDEMIC_H
#define MEPIDEMIC_H

// Compute the number of transmission events from humans to mosquitoes
// across all sub-populations.
__global__ void mVisitingInfected(curandState_t* d_randStates,
                                  uint32_t* d_visitInfected,
                                  const uint32_t* d_infectedCount,
                                  const float* d_sparseNetWeight,
                                  const uint32_t* d_sparseNetTo,
                                  const uint32_t* d_sparseNetLoc,
                                  const uint32_t* d_subPopTotal,
                                  const Parameter h_parameter);

// Infects mosquitoes given the number of transmission events from humans to
// mosquitoes in each subpopulation for each strain.
__global__ void nmTransmission(age* d_mAge,
                               dead* d_mDead,
                               exposed* d_mExposed,
                               infectStatus* d_mInfectStatus,
                               strain* d_mStrain,
                               curandState_t* d_randStates,
                               const uint32_t* d_subPopIndex,
                               const uint32_t* d_subPopLoc,
                               const uint32_t* d_subPopSize,
                               const uint32_t* d_nTransmission,
                               const uint32_t* d_mSubPopCount,
                               const uint32_t* d_size,
                               const uint32_t* d_subPopTotal,
                               const uint32_t t,
                               const Parameter h_parameter);

#endif
