// epidemicinitial.cu: this CUDA file contains the host functions and device kernels
// for the initialization of infection into the human and mosquito populations.

#include "constant.h"                   // Constants for the simulation.
#include "censustypedef.h"              // Type definitions for census data.
#include "curand_kernel.h"              // CUDA random number generation.
#include "cudaidentities.h"             // Warp ID, streaming multiprocessor ID and laneID.
#include "device_launch_parameters.h"   // Block ID and thread ID.
#include "Parameter.h"                  // Parameter space definition.

// Device function which determines if a float should be rounded up or down, dependent
// on uniform random number generation.
__device__ __forceinline__ uint32_t decimalResolve(curandState_t* state, float floating)
{
    float frac = floating - floorf(floating);
    if (curand_uniform(state) <= frac)
    {
        return static_cast<uint32_t>(ceilf(floating));
    }
    else
    {
        return static_cast<uint32_t>(floorf(floating));
    }
}

// CUDA kernel initializing the infection in the human population.
__global__ void nEpidemicInitial(age* d_nAge,
                                 history* d_nHistory,
                                 infectStatus* d_nInfectStatus,
                                 recovery* d_nRecovery,
                                 strain* d_nStrain,
                                 subPopulation* d_nSubPopulation,
                                 curandState_t* d_randStates,
                                 uint32_t* d_nInfectedCount,
                                 const uint32_t* d_nSize,
                                 const uint32_t* d_subPopTotal,
                                 const uint32_t nInitialInfected,
                                 const uint32_t nGridSize,
                                 const Parameter h_parameter)
{
    // Initialize the human index for census data access.
    uint32_t nIndex = blockIdx.x*blockDim.x + threadIdx.x;

    // Determine which active thread is being run (for random number generation).
    uint32_t activeThreadId = smId()*h_parameter.arch.warpsPerSM*h_parameter.arch.threadsPerWarp
        + warpId()*h_parameter.arch.threadsPerWarp + laneId();

    // Infect the first nInitialInfected humans in the census.
    if (nIndex < nInitialInfected)
    {
        // Load the random number generator state from global memory.
        curandState_t local_state = d_randStates[activeThreadId];
        
        // Randomly assign the individual on the census a strain of the virus. Alter the individuals
        // immunological history, and the age at which they recover from the infection.
        d_nInfectStatus[nIndex] = 2;
        strain local_strain = static_cast<strain>(curand(&local_state) % C_STRAINS);
        d_nStrain[nIndex] = local_strain;
        d_nHistory[local_strain*d_nSize[0] + nIndex] = -2;

        // Age at which individual stops being infected.
        recovery recoveryTime = curand(&local_state) % (decimalResolve(&local_state, h_parameter.simulation.recovery) - 1);
        d_nRecovery[nIndex] = static_cast<recovery>(recoveryTime + 1);

        // Increment the number of infected individuals of that strain in the individuals sub-population.
        atomicAdd(&d_nInfectedCount[local_strain* d_subPopTotal[0] + d_nSubPopulation[nIndex]], 1);
    }
}

// CUDA kernel initializing infection in the mosquito population.
__global__ void mEpidemicInitial(infectStatus* d_mInfectStatus,
                                 strain* d_mStrain,
                                 subPopulation* d_mSubPopulation,
                                 curandState_t* d_randStates,
                                 uint32_t* d_mInfectedCount,
                                 const uint32_t* d_subPopTotal,
                                 const uint32_t mInitialInfected,
                                 const uint32_t mGridSize,
                                 const ArchitectureParam h_archParam)
{
    // Initialize the mosquito index for census data access.
    uint32_t mIndex = blockIdx.x*blockDim.x + threadIdx.x;

    // Determine which active thread is being run (for random number generation).
    uint32_t activeThreadId = smId()*h_archParam.warpsPerSM*h_archParam.threadsPerWarp
        + warpId()*h_archParam.threadsPerWarp + laneId();

    // Infect the first mInitialInfected mosquitoes in the census.
    if (mIndex < mInitialInfected)
    {
        // Randomly assign the individual on the census a strain of the virus.
        d_mInfectStatus[mIndex] = 2;
        strain local_strain = static_cast<strain>(curand(&d_randStates[activeThreadId]) % C_STRAINS);
        d_mStrain[mIndex] = local_strain;

        // Increment the number of infected individuals of that strain in the individuals sub-population.
        atomicAdd(&d_mInfectedCount[local_strain*d_subPopTotal[0] + __ldg(&d_mSubPopulation[mIndex])], 1);
    }
}

// CUDA Kernel which intiializes the infected counters for each subpopulation for each species.
__global__ void infectCountInitial(uint32_t* d_nInfectedSubPopCount,
                                   uint32_t* d_mInfectedSubPopCount,
                                   const uint32_t* d_subPopTotal)
{
    // Determine the strain, subpopulation combination the thread is responsible for.
    uint32_t strainSubPop = threadIdx.x + blockIdx.x*blockDim.x;

    // Ensure the combination is well defined.
    if (strainSubPop < d_subPopTotal[0]*C_STRAINS)
    {
        d_nInfectedSubPopCount[strainSubPop] = 0;
        d_mInfectedSubPopCount[strainSubPop] = 0;
    }
}

// Reset the daily incidence of each strain in each sub-population to zero.
__global__ void initialIncidence(uint32_t* d_nIncidenceSubPop,
    const uint32_t* d_subPopTotal)
{
    // Get the one-dimensional index of the (strain, subpopulation) combination.
    uint32_t strainSubPopIndex = threadIdx.x + blockDim.x * blockIdx.x;

    // If the one-dimensional index is valid, reset the daily incidence of that (strain, subpopulation)
    // combination to zero.
    if (strainSubPopIndex < __ldg(&d_subPopTotal[0]) * C_STRAINS)
    {
        d_nIncidenceSubPop[strainSubPopIndex] = 0;
    }
}

// Host function which calls CUDA kernels to initial the epidemics of the
// disease in the human and mosquito populations and initialize the counter for
// all strain, subpopulation combinations.
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
                              const Parameter h_parameter)
{
    // Determine the size of the grid of blocks required on the GPU to reset daily incidence.
    uint32_t resetGridSize = static_cast<uint32_t>(ceil(h_parameter.simulation.metaPop * C_STRAINS / static_cast<float>(C_THREADSPERBLOCK)));

    // Reset the daily incidence counter fo each strain in each sub-population to zero, ready
    // to count each new case for that day.
    initialIncidence << < resetGridSize, C_THREADSPERBLOCK >> >(d_nIncidenceSubPop, d_subPopTotal);

    // Initialize the infected counters.
    uint32_t infectSubPopGridSize{ static_cast<uint32_t>(ceil(C_STRAINS*h_subPopTotal /
        static_cast<float>(C_THREADSPERBLOCK))) };
    infectCountInitial<<< infectSubPopGridSize, C_THREADSPERBLOCK >>>
        (d_nInfectedSubPopCount, d_mInfectedSubPopCount, d_subPopTotal);

    // Determine the grid size to use on the GPU for epidemic initialization with a pre-determined number of threads
    // per block on the grid (must be a whole number).
    uint32_t nInfectGridSize{ static_cast<uint32_t>(ceil(nInitialInfected / 
                                                        static_cast<float>(C_THREADSPERBLOCK))) };
    uint32_t mInfectGridSize{ static_cast<uint32_t>(ceil(mInitialInfected / 
                                                        static_cast<float>(C_THREADSPERBLOCK))) };

    // Infect the first nInitialInfected/mInitialInfected humans/mosquitoes in their respective census data.
    // Keep track of the number of infected individuals for each subpopulation and virus strain combination.
    nEpidemicInitial<<< nInfectGridSize, C_THREADSPERBLOCK >>>
        (d_nAge, d_nHistory, d_nInfectStatus, d_nRecovery, d_nStrain, d_nSubPopulation, 
        d_randStates, d_nInfectedSubPopCount, d_nSize, d_subPopTotal, nInitialInfected, nGridSize, h_parameter);
    mEpidemicInitial<<< mInfectGridSize, C_THREADSPERBLOCK >>>
        (d_mInfectStatus, d_mStrain, d_mSubPopulation, 
        d_randStates, d_mInfectedSubPopCount, d_subPopTotal, mInitialInfected, mGridSize, h_parameter.arch);
}