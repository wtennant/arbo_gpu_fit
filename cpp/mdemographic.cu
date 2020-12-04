// mdemographics.cu: This file contains the host code which invokes CUDA kernels 
// for the simulation of the mosquito demographics.

#include "censustypedef.h"              // Type definitions for census data.
#include "constant.h"                   // Constants for simulation.
#include "curand_kernel.h"              // Required for random number generation in CUDA.
#include "cudaidentities.h"             // Streaming multiprocessor ID, warp ID, and lane ID.
#include "device_launch_parameters.h"   // Thread ID and block ID.
#include "Parameter.h"                  // Parameter space definition.

// The cumulative life expectancy Weibull distribution for mosquitos. This distribution
// gives the most flexibility in age-specific mortality rates.
__device__ __forceinline__ float device_mLifeExpectancy(const age local_age, const SimulationParam h_simParam)
{
    // The cumulative weibull distribution is given by 1 - exp(-(x / scale)^shape)
    return (1.0f - expf(-(powf(local_age / h_simParam.mScaleLifeExpectancy, h_simParam.mShapeLifeExpectancy))));
}

// Determines if an individual dies at the current time step and if a deceased individual 
// needs to be replaced by a new individual. Keeps track of the number of dead individuals
// on the GPU block.
__global__ void mBlockBirthDeath(age* d_mAge,
                                 dead* d_mDead,
                                 exposed* d_mExposed,
                                 infectStatus* d_mInfectStatus,
                                 pLifeExpectancy* d_mPLE,
                                 strain* d_mStrain,
                                 subPopulation* d_mSubPopulation,
                                 curandState_t* d_randStates,
                                 uint32_t* d_infectedCount,
                                 uint32_t* d_subPopCount,
                                 uint16_t* d_deadCount,
                                 const uint32_t* d_size,
                                 const uint32_t* d_subPopTotal,
                                 const float* d_mExpectedPopSize,
                                 const Parameter h_parameter)
{
    // Initialize the mosquito index for the thread.
    uint32_t mIndex = threadIdx.x + blockDim.x * blockIdx.x;

    // Initialize the active thread index for random number generation.
    uint32_t activeThreadId = smId()*h_parameter.arch.warpsPerSM*h_parameter.arch.threadsPerWarp
        + warpId()*h_parameter.arch.threadsPerWarp + laneId();

    // Declare the shared variables and fast read in the current population size with __ldg.
    __shared__ uint32_t shared_deadCount;         // The number of dead individuals on the GPU block.
    __shared__ int32_t shared_extraBirthCount;    // The number of new birth required on the GPU block.
    uint32_t local_size = __ldg(d_size);

    // Initialize the dead counter shared variables.
    if (threadIdx.x == 0)
    {
        shared_deadCount = d_deadCount[blockIdx.x];
    }

    // Ensure shared memory has been initialized before proceeding.
    __syncthreads();

    // Initialize the varaible which accounts for the net birth/death ratio of the thread.
    // This saves on atomic incrementation of subpopulation count data.
    int32_t netThreadBirthDeath{ 0 };

    // If the individual is well defined, update the individuals demographics.
    if ((mIndex < local_size) && (!d_mDead[mIndex]))
    {
		// Load in the individuals age and if they are susceptible, infected, or infectious.
		age local_age = (++d_mAge[mIndex]);
        infectStatus local_infectStatus = d_mInfectStatus[mIndex];

        // Determine if the mosquito dies from the cumulative life expectancy distribution and the 
        // individuals pre-determined life expectancy probability.
        if (__ldg(&d_mPLE[mIndex]) <= device_mLifeExpectancy(local_age, h_parameter.simulation))
        {
            // If the mosquito was infected with a virus when they died, decrease counter containing the total 
            // number of infected individuals for the subpopulation they belong to, for that strain.
            if ((local_infectStatus == 2) && (!d_mDead[mIndex]))
            {
                atomicSub(&d_infectedCount[d_mStrain[mIndex] * __ldg(&d_subPopTotal[0]) + __ldg(&d_mSubPopulation[mIndex])], 1);
            }

            // Record that the individual is dead.
            d_mDead[mIndex] = 1;
            --netThreadBirthDeath;

            // Increase the counter for dead individuals on the block.
            atomicAdd(&shared_deadCount, 1);
        }
        else
        {
            // If the individual is infected but not infectious, determine if the individual
            // needs to be come infectious.
            if ((local_infectStatus == 1) && (d_mExposed[mIndex] < local_age))
            {
                atomicAdd(&d_infectedCount[d_mStrain[mIndex] * __ldg(&d_subPopTotal[0]) + __ldg(&d_mSubPopulation[mIndex])], 1);
                d_mInfectStatus[mIndex] = 2;
            }
        }
    }

    // Ensure that all necessary deaths have occured at the time step before
    // computing the number of births required in order to give the correct population size.
    __syncthreads();

    // Compute the number of births needed to give the correct population size of the block. 
    if (threadIdx.x == 0)
    {
        // Determine the maximum number of individuals on the block.
        float extraBirthCount = fminf(local_size - mIndex, blockDim.x);

        // Determine the number of births required on the block in order to get the expected population size of the block.
        extraBirthCount = (__ldg(&d_mExpectedPopSize[0])*extraBirthCount) / static_cast<float>(local_size)
            - extraBirthCount + shared_deadCount;

        // Get the fractional part of the expected number of births on the block, and randomly generate a probability in
        // order to determine if the expected number of births should be rounded up or down.
        float frac_extraBirthCount = extraBirthCount - floorf(extraBirthCount);
        if (curand_uniform(&d_randStates[activeThreadId]) <= frac_extraBirthCount)
        {
            shared_extraBirthCount = static_cast<int32_t>(ceilf(extraBirthCount));
        }
        else
        {
            shared_extraBirthCount = static_cast<int32_t>(floorf(extraBirthCount));
        }

        // All births are guaranteed to occur, so decrease the number of dead individuals on the block
        // and store this back to global memory.
        d_deadCount[blockIdx.x] = shared_deadCount - shared_extraBirthCount;
    }

    // The extra number of births required at the current timestep is required on all threads in the block before continuing.
    __syncthreads();

    // If the individual is dead and a birth is required, birth a new individual.
    if ((mIndex < local_size) && (d_mDead[mIndex]) && (atomicSub(&shared_extraBirthCount, 1) > 0))
    {
        d_mAge[mIndex] = 0;
        d_mDead[mIndex] = 0;
        d_mInfectStatus[mIndex] = 0;
        d_mPLE[mIndex] = curand_uniform(&d_randStates[activeThreadId]);
        ++netThreadBirthDeath;
    }

    // If an individual only died, or was only birthed during the current time step, record
    // this change in the subpopulation size.
    if (netThreadBirthDeath != 0)
    {
        atomicAdd(&d_subPopCount[d_mSubPopulation[mIndex]], netThreadBirthDeath);
    }
}

// The CPU code which calls all CUDA kernels involved in the update of mosquito demographics.
__host__ void mDemographic(age* d_mAge,
                           dead* d_mDead,
                           exposed* d_mExposed,
                           infectStatus* d_mInfectStatus,
                           pLifeExpectancy* d_mPLE,
                           strain* d_mStrain,
                           subPopulation* d_mSubPopulation,
                           curandState_t* d_randStates,
                           uint32_t* d_infectedCount,
                           uint32_t* d_subPopCount,
                           uint16_t *d_deadCount,
                           uint32_t* d_size,
                           const uint32_t* d_subPopTotal,
                           const uint32_t sizeReserve,
                           const float* d_mExpectedPopSize,
                           const Parameter h_parameter)
{
    // Determine the size of the grid of blocks required on the GPU to do the update of mosquito demographics.
    uint32_t gridSize = static_cast<uint32_t>(ceil(sizeReserve / static_cast<float>(C_THREADSPERBLOCK)));

    // Call the kernel which determines if a mosquito dies or not, and if they are to be replaced by a new individual.
    mBlockBirthDeath<<< gridSize, C_THREADSPERBLOCK >>>
        (d_mAge, d_mDead, d_mExposed, d_mInfectStatus, d_mPLE, d_mStrain, d_mSubPopulation, d_randStates,
        d_infectedCount, d_subPopCount, d_deadCount, d_size, d_subPopTotal, d_mExpectedPopSize, h_parameter);
}