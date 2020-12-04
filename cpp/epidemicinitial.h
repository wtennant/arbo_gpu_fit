// epidemicinitial.h: contains the function declaration for initialization an epidemic
// in the meta-population.

#include "censustypedef.h"      // Type definitions for census data.
#include "host_defines.h"       // Allows definition of host functions.
#include "curand_kernel.h"      // CUDA random number generator type.
#include "Parameter.h"          // Parameter space definition.

// Start of the header guard.
#ifndef EPIDEMICINITIAL_H
#define EPIDEMICINITIAL_H

// Host function that intializes infection among the human and mosquito population.
__host__ void epidemicInitial(age* d_nAge,
                              history* d_nHistory,
                              infectStatus* d_nInfectStatus,
                              recovery* d_nRecovery,
                              strain* d_nStrain,
                              subPopulation* d_nSubPopulation,
                              infectStatus* d_mInfectStatus,
                              strain* d_mStrain,
                              subPopulation* d_mSubPopulation,
                              curandState_t* d_randStates,
                              uint32_t* d_nInfectedSubPopCount,
                              uint32_t* d_mInfectedSubPopCount, 
                              uint32_t* d_nIncidenceSubPop,
                              const uint32_t* d_nSize,
                              const uint32_t* d_subPopTotal,
                              const uint32_t nInitialInfected,
                              const uint32_t mInitialInfected,
                              const uint32_t nGridSize,
                              const uint32_t mGridSize,
                              const uint32_t h_subPopTotal,
                              const Parameter h_parameter);

#endif