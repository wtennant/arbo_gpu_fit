// This header file contains the declarations for the host function
// that invokes CUDA kernels to reduce infected and population counts
// across all sub-populations, and stores the result in a time series.

#include "host_defines.h"   // Definition of host-only functions.

// Start of the header guard.
#ifndef DATACOLLECT_H
#define DATACOLLECT_H

// Host function with invokes CUDA kernels to first reduce infection and population
// counts across all sub-populations, then stores the results in time-series arrays
// on the device.
__host__ void dataCollect(uint32_t* d_nReductionInfectedCount,
                          uint32_t* d_nReductionCount,
                          uint32_t* d_mReductionCount,
                          uint32_t* d_nInfectedSubPopCountSeries,
                          uint32_t* d_nOneSubPopInfectedCount,
                          uint32_t* d_nInfectedCount,
						  uint32_t* d_nIncidence,
                          uint32_t* d_nCount,
                          uint32_t* d_mCount,
                          const uint32_t* d_nInfectedSubPopCount,
						  const uint32_t* d_nIncidenceSubPop,
                          const uint32_t* d_nSubPopCount,
                          const uint32_t* d_mSubPopCount,
                          const uint32_t h_specialSubPop,
                          const uint32_t h_subPopTotal,
                          const uint32_t t);

#endif