// Header file containing declarations for host functions that invoke CUDA kernels
// to initialize and update the human and mosquito demographics (deaths, births) as well as
// the transitions between different infectivity classes (susceptible, exposed, infectious, recovered).

#include "censustypedef.h"      // Type definitions for census data.
#include "host_defines.h"       // Allows definition of host functions.
#include "curand_kernel.h"      // CUDA random number generator type.
#include "Parameter.h"          // Parameter space definition.

// Start of the header guard.
#ifndef DEMOGRAPHIC_H
#define DEMOGRAPHIC_H

// Host function which invokes CUDA kernel for updating ages of human individuals,
// determining if individuals need to be replaced (constant population), and
// if an infected individual needs to become infectious/recovered.
__host__ void nDemographic(age* d_nAge,
                           exposed* d_nExposed,
                           history* d_nHistory,
                           infectStatus* d_nInfectStatus,
                           pLifeExpectancy* d_nPLE,
                           recovery* d_nRecovery,
                           strain* d_nStrain,
                           subPopulation* d_nSubPopulation,
                           curandState_t* d_randStates,
                           uint32_t* d_nInfectedCount,
						   uint32_t* d_nIncidenceSubPop,
                           uint32_t* d_nSize,
                           const uint32_t* d_subPopTotal,
                           const uint32_t h_nSize,
                           const Parameter h_parameter);

// Host function which invokes CUDA kernel for updating ages of mosquito individuals,
// determining death of individuals, birth of individuals (seasonal), and 
// transitions between susceptible, exposed, and infected (no recovery).
__host__ void mDemographic(age* d_mAge,
                           dead* d_nDead,
                           exposed* d_nExposed,
                           infectStatus* d_nInfectStatus,
                           pLifeExpectancy* d_nPLE,
                           strain* d_nStrain,
                           subPopulation* d_nSubPopulation,
                           curandState_t* d_randStates,
                           uint32_t* d_infectedCount,
                           uint32_t* d_subPopCount,
                           uint16_t *d_deadCount,
                           uint32_t* d_size,
                           const uint32_t* d_subPopTotal,
                           const uint32_t sizeReserve,
                           const float* d_mExpectedPopSize,
                           const Parameter h_parameter);

#endif