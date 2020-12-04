// demographicinitial.h: contains the declaration of intiailizing the
// demography of the vector and host populations.

#include "censustypedef.h"      // Type definitions for census data.
#include "host_defines.h"       // Allows definition of host functions.
#include "curand_kernel.h"      // CUDA random number generator type.
#include "Parameter.h"          // Parameter space definition.

#ifndef DEMOGRAPHICINITIAL_H
#define DEMOGRAPHICINITIAL_H

// Initializes the human and mosquito populations (without disease).
__host__ void demographicInitialization(age* d_nAge,
                                        history* d_nHistory,
                                        infectStatus* d_nInfectStatus,
                                        pLifeExpectancy* d_nPLE,
                                        subPopulation* d_nSubPopulation,
                                        age* d_mAge,
                                        dead* d_mDead,
                                        infectStatus* d_mInfectStatus,
                                        pLifeExpectancy* d_mPLE,
                                        subPopulation* d_mSubPopulation,
                                        uint32_t* d_nSubPopIndex,
                                        uint32_t* d_mSubPopIndex,
                                        curandState_t* d_randStates,
                                        uint32_t* d_nSubPopCount,
                                        uint32_t* d_mSubPopCount,
                                        uint32_t* d_nSubPopLoc,
                                        uint32_t* d_mSubPopLoc,
                                        uint32_t* d_nSubPopSize,
                                        uint32_t* d_mSubPopSize,
                                        uint16_t* d_mDeadCount,
                                        float* d_nSurvival,
                                        float* d_mSurvival,
                                        const float* d_nCumulativeComSize,
                                        const float* d_mCumulativeComSize,
                                        const float* d_mExpectedPopSize,
                                        const uint32_t* d_nSize,
                                        const uint32_t* d_mSize,
                                        const uint32_t* d_subPopTotal,
                                        const uint32_t h_nSize,
                                        const uint32_t h_mSizeReserve,
                                        const uint32_t h_subPopTotal,
                                        const Parameter h_parameter);

#endif