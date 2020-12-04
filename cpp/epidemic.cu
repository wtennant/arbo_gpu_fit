// epidemic.cu: contains the host function which calls the CUDA disease transmission kernels.

#include "censustypedef.h"              // Type definitions for census data.
#include "curand_kernel.h"              // Device functions for CUDA random number generation.
#include "constant.h"                   // Constants for the simulation.
#include "nepidemic.h"                  // To human transmission.
#include "mepidemic.h"                  // To mosquito transmission.
#include "device_launch_parameters.h"   // Thread ID and block ID.

// CUDA kernel which intializes the number of transmission events between species
// for every subpopulation and strain.
__global__ void initialTransmission(uint32_t* d_nTransmission,
                                    uint32_t* d_mTransmission,
                                    const uint32_t* d_subPopTotal)
{
    // Initialize the strain, subpopulation combination of the thread.
    uint32_t strainSubPop = blockIdx.x*blockDim.x + threadIdx.x;

    // Load in the total number of subpopulation from global memory to L1 cache.
    uint32_t subPopTotal = __ldg(&d_subPopTotal[0]);

    // Initialize the number of transmission events that occur to individuals of every subpopulation
    // and strain combination.
    if (strainSubPop < subPopTotal*C_STRAINS)
    {
        d_nTransmission[strainSubPop] = 0;
        d_mTransmission[strainSubPop] = 0;
    }
}

// Host function which calls the CUDA kernels for simulating disease transmission.
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
                       const Parameter h_parameter)
{
    // Call the CUDA kernel which initializes transmission numbers.
    uint32_t infectedGridSize = static_cast<uint32_t>(ceil(h_subPopTotal*C_STRAINS / static_cast<float>(C_THREADSPERBLOCK)));
    initialTransmission<<< infectedGridSize, C_THREADSPERBLOCK >>>
        (d_nTransmission, d_mTransmission, d_subPopTotal);

    // Call the CUDA kernel which determines transmission numbers from each sub-population to another subpopulation.
    nVisitingInfected<<< infectedGridSize, C_THREADSPERBLOCK >>>
        (d_randStates, d_nTransmission, d_nInfectedSubPopCount, d_nSubPopCount, d_mSubPopCount, d_sparseNetWeight, d_sparseNetTo, d_sparseNetLoc, d_subPopTotal, h_parameter);
    mVisitingInfected<<< infectedGridSize, C_THREADSPERBLOCK >>>
        (d_randStates, d_mTransmission, d_mInfectedSubPopCount, d_sparseNetWeight, d_sparseNetTo, d_sparseNetLoc, d_subPopTotal, h_parameter);

    // Call the CUDA kernel which infects human and mosquitoes based on the transmission numbers computed
    // in the previous kernel invokation.
    uint32_t subPopGridSize = static_cast<uint32_t>(ceil(h_subPopTotal / static_cast<float>(C_THREADSPERBLOCK)));
    nmTransmission<<< subPopGridSize, C_THREADSPERBLOCK >>>
        (d_mAge, d_mDead, d_mExposed, d_mInfectStatus, d_mStrain, d_randStates,
            d_mSubPopIndex, d_mSubPopLoc, d_mSubPopSize, d_nTransmission, d_mSubPopCount, d_mSize, d_subPopTotal, t, h_parameter);
    mnTransmission<<< subPopGridSize, C_THREADSPERBLOCK >>>
        (d_nAge, d_nExposed, d_nHistory, d_nInfectStatus, d_nRecovery, d_nStrain, d_randStates, d_nAgeOfInfection, d_nAgeOfInfectionCount,
            d_nInfectedCount, d_nSubPopIndex, d_nSubPopLoc, d_nSubPopSize, d_mTransmission, d_nSize, d_subPopTotal, t, maxTime, h_parameter);
}