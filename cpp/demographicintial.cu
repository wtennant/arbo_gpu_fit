// demographicintial.cu: host function and CUDA kernels for the intialization of the human 
// and mosquito population (disease-free).

#include <cstdint>                      // Fixed width integers.
#include "curand_kernel.h"              // Device functions for CUDA random number generation.
#include "cudaidentities.h"             // Identities for determining the SM, warp, and lane id of a GPU thread. 
#include "device_launch_parameters.h"   // CUDA block id, thread id.
#include "constant.h"                   // Constants for simulation.
#include "censustypedef.h"              // Type definitions for census data.
#include "Parameter.h"                  // Parameter space definition.

// The cumulative life expectancy bi-Weibull distribution for humans. This distribution
// gives the most flexibility in age-specific mortality rates.
static __forceinline__ __device__ float nLifeExpectancy(age local_age, SimulationParam h_simParam)
{
    // Initialize as the argument of the burn-in exponential.
    float cumulativeLifeExpectancy = powf(static_cast<float>(local_age * h_simParam.nScaleInfantMortality), h_simParam.nShapeInfantMortality);

    // Need more terms if beyond the location where the second (decay out) Weibull distribution begins.
    if (local_age < h_simParam.nLocWeibull)
    {
        cumulativeLifeExpectancy = 1 - exp(-cumulativeLifeExpectancy);
    }
    else
    {
        cumulativeLifeExpectancy = cumulativeLifeExpectancy +
            powf(static_cast<float>(local_age - h_simParam.nLocWeibull) / h_simParam.nScaleLifeExpectancy, h_simParam.nShapeLifeExpectancy);
        cumulativeLifeExpectancy = 1 - exp(-cumulativeLifeExpectancy);
    }

    return cumulativeLifeExpectancy;
}

// The cumulative life expectancy Weibull distribution for mosquitos. This distribution
// gives the most flexibility in age-specific mortality rates.
static __forceinline__ __device__ float mLifeExpectancy(age local_age, SimulationParam h_simParam)
{
    // The cumulative weibull distribution is given by 1 - exp(-(x / scale)^shape)
    float cumulativeLifeExpectancy{ 1 - exp(-powf(local_age / h_simParam.mScaleLifeExpectancy, 
        h_simParam.mShapeLifeExpectancy)) };

    return cumulativeLifeExpectancy;
}

// The CUDA kernel for the human cumulative survival function. This gives an 
// estimate for the expected value of the human life expectancy distribution.
// Upper Rieman Sum of the survival function, overestimates expected value.
__global__ void nCumulativeSurvival(float* d_nSurvival, SimulationParam h_simParam)
{
    // Define the first probability of surviving (at age 0 days).
    d_nSurvival[0] = static_cast<float>(1 - nLifeExpectancy(0, h_simParam));

    // Initialize the age counter (in years)
    uint16_t ageCount = 1;

    // For every age, calculate the probability of surviving up until that age,
    // and sum with all probabilities of surviving up until all previous ages.
    while (ageCount <= C_NMAXINITIALAGE)
    {
        d_nSurvival[ageCount] = d_nSurvival[ageCount - 1] + (1 - nLifeExpectancy(ageCount*C_YEAR, h_simParam));
        ++ageCount;
    }
}

// The CUDA kernel for the human cumulative survival function. This gives an 
// estimate for the expected value of the mosquito life expectancy distribution.
// Upper Rieman Sum of the survival function, overestimates expected value.
__global__ void mCumulativeSurvival(float* d_mSurvival, SimulationParam h_simParam)
{
    // Define the first probability of surviving (at age 0 days).
    d_mSurvival[0] = static_cast<float>(1 - mLifeExpectancy(0, h_simParam));

    // Initialize the age counter (in days)
    uint16_t ageCount = 1;

    // For every age, calculate the probability of surviving up until that age,
    // and sum with all probabilities of surviving up until all previous ages.
    while (ageCount <= C_MMAXINITIALAGE)
    {
        d_mSurvival[ageCount] = d_mSurvival[ageCount - 1] + (1 - mLifeExpectancy(ageCount, h_simParam));
        ++ageCount;
    }
}

// The CUDA kernel for the initialization of the human population.
__global__ void nDemographicInitialization(age* d_nAge,
                                           history* d_nHistory,
                                           infectStatus* d_nInfectStatus,
                                           pLifeExpectancy* d_nPLE,
                                           curandState_t* d_randStates,
                                           const float* d_nSurvival,
                                           const uint32_t* d_nSize,
                                           const Parameter h_parameter)
{
    // Initalize the human index which determines the human in the census data
    // that is assigned to the GPU thread.
    uint16_t threadid = threadIdx.x;
    uint32_t blockid = blockIdx.x;
    uint32_t nIndex = blockid*blockDim.x + threadid;

    // Determine which active thread the individual is being run on. This is used
    // for random number generation.
    uint32_t activeThreadId = smId()*h_parameter.arch.warpsPerSM*h_parameter.arch.threadsPerWarp
        + warpId()*h_parameter.arch.threadsPerWarp + laneId();

    // Create a human if the human index is within the requested size of the human
    // population.
    uint32_t local_nSize = __ldg(&d_nSize[0]);
    if (nIndex < local_nSize)
    {
        // Load the random number generator state from global memory to local memory.
        curandState_t localState = d_randStates[activeThreadId];

        // Declare the age variable for the human.
        age local_age;

        // Determine the age of an individual over the bi-Weibull distribution's cumulative survival function.
        // Choose a random value between zero and the final cumulative survival function.
        float randCSF = d_nSurvival[C_NMAXINITIALAGE] * curand_uniform(&localState);

        // Starting at age zero, scan through the cumulative survival function and determine the age that
        // the individual has survived until. Once an age has been assigned, exit the loop.
        // In other words, determine what age in the cumulative survival function gives randCSF.
        uint16_t ageCount = 0;
        while (ageCount <= C_NMAXINITIALAGE)
        {
            if (randCSF <= d_nSurvival[ageCount])
            {
                // Randomly assign a day of their birth within the year they were born.
                local_age = static_cast<uint16_t>(C_YEAR*ageCount + (curand(&localState) % C_YEAR));
                break;
            }
            else
            {
                ++ageCount;
            }
        }

        // Assign a life expectancy probability to an individual such that it is greater than the life 
        // expectancy probability of their current age (scale, and move up uniform disitribution). However,
        // to avoid the small blip in the first run through of the simulation, where no-one dies, step age
        // back by one when calculating the life probability (unless already zero).
        if (local_age != 0)
        {
            d_nPLE[nIndex] = (1 - nLifeExpectancy(local_age - 1, h_parameter.simulation)) * curand_uniform(&localState) +
                nLifeExpectancy(local_age - 1, h_parameter.simulation);
        }
        else
        {
            d_nPLE[nIndex] = (1 - nLifeExpectancy(local_age, h_parameter.simulation)) * curand_uniform(&localState) +
                nLifeExpectancy(local_age, h_parameter.simulation);
        }

        // Initially assume that all individuals are not infected.
        d_nInfectStatus[nIndex] = 0;

        // Set some kind of immunity profile in the human population.
		uint8_t strain{ 0 };
		while (strain < C_STRAINS)
		{
            if (curand_uniform(&localState) < h_parameter.simulation.initialSeroPrev)
            {
                // History of an infection from initialization is indicated with -2.
                d_nHistory[strain*local_nSize + nIndex] = -2;
            }
            else
            {
                // No history is indicated with a value of -1.
                d_nHistory[strain*local_nSize + nIndex] = -1;
            }
			++strain;
        }

        // Record the random number generator state back to global memory.
        d_randStates[activeThreadId] = localState;

        // Record the human from local memory to global memory.
        d_nAge[nIndex] = local_age;
    }
}

// The CUDA kernel for the initialization of the mosquito population.
__global__ void mDemographicInitialization(age* d_mAge,
                                           dead* d_mDead,
                                           infectStatus* d_mInfectStatus,
                                           pLifeExpectancy* d_mPLE,
                                           subPopulation* d_mSubPopulation,
                                           curandState_t* d_randStates,
                                           uint16_t* d_mDeadCount,        
                                           uint32_t* d_mSubPopCount,
                                           const float* d_mSurvival,
                                           const float* d_mExpectedPopSize,
                                           const uint32_t* d_mSize,
                                           const Parameter h_parameter)
{
    // Initalize the mosquito index which determines the human in the census data
    // that is assigned to the GPU thread.
    uint16_t threadIndex = threadIdx.x;
    uint32_t blockIndex = blockIdx.x;
    uint32_t mIndex = blockIndex*blockDim.x + threadIndex;

    // Determine which active thread the individual is being run on. This is used
    // for random number generation.
    uint32_t activeThreadId = smId()*h_parameter.arch.warpsPerSM*h_parameter.arch.threadsPerWarp
        + warpId()*h_parameter.arch.threadsPerWarp + laneId();

    // Declare the shared variables.
    __shared__ uint32_t shared_deadCount;         // The number of dead individuals on the GPU block.

    // Initialize the counter for the total number of dead individuals on a block. 
    if (threadIndex == 0)
    {
        shared_deadCount = 0;
    }

    // Synchronize threads across a block.
    __syncthreads();

    // Create a human if the human index is within the requested size of the human
    // population.
    if (mIndex < d_mSize[0])
    {
        // Load the random number generator state from global memory to local memory.
        curandState_t localState = d_randStates[activeThreadId];

        // Determine if the individual is alive or not.
        if (curand_uniform(&localState) >= d_mExpectedPopSize[0]/d_mSize[0])
        {
            d_mDead[mIndex] = 1;
            atomicAdd(&shared_deadCount, 1);
            atomicSub(&d_mSubPopCount[d_mSubPopulation[mIndex]], 1);
        }
        else
        {
            // Determine the age of an individual over the bi-Weibull distribution's cumulative survival function.
            // Choose a random value between zero and the final cumulative survival function.
            float randCSF = d_mSurvival[C_MMAXINITIALAGE] * curand_uniform(&localState);

            // Starting at age zero, scan through the cumulative survival function and determine the age that
            // the individual has survived until. Once an age has been assigned, exit the loop.
            // In other words, determine what age in the cumulative survival function gives randCSF.
            age local_age;
            uint16_t ageCount = 0;
            bool ageAssigned = false;
            while ((ageCount <= C_MMAXINITIALAGE) && (!ageAssigned))
            {
                if (randCSF <= d_mSurvival[ageCount])
                {
                    // Assign the age which the mosquito has thus far survived until.
                    local_age = ageCount;
                    ageAssigned = true;
                }
                else
                {
                    ++ageCount;
                }
            }

            // Individuals are alive when created.
            d_mDead[mIndex] = 0;

            // Assign a life expectancy probability to an individual such that it is greater than the life 
            // expectancy probability of their current age (scale, and move up uniform disitribution). However,
            // to avoid the small blip in the first run through of the simulation, where no-one dies, step age
            // back by one when calculating the life probability (unless already zero).
            if (local_age != 0)
            {
                d_mPLE[mIndex] = (1 - mLifeExpectancy(local_age - 1, h_parameter.simulation)) * curand_uniform(&localState) +
                    mLifeExpectancy(local_age - 1, h_parameter.simulation);
            }
            else
            {
                d_mPLE[mIndex] = (1 - mLifeExpectancy(local_age, h_parameter.simulation)) * curand_uniform(&localState) +
                    mLifeExpectancy(local_age, h_parameter.simulation);
            }

            // Initially assume that all individuals are not infected. Not necessary to set a strain.
            d_mInfectStatus[mIndex] = 0;

            // Record the mosquito from local memory to global memory.
            d_mAge[mIndex] = local_age;
        }

        // Record the random number generator state back to global memory.
        d_randStates[activeThreadId] = localState;
    }

    // Ensure that all necessary deaths have occured.
    __syncthreads();

    // Save the number of dead mosquitoes on the block. Required for
    // mosquito demographic simulation.
    if (threadIndex == 0)
    {
        d_mDeadCount[blockIdx.x] = shared_deadCount;
    }
}

// CUDA kernel which computes how many individuals are in each subpopulation 
// (reserved individuals and initialized individuals).
__global__ void subPopCount(uint32_t* d_nSubPopCount,
                            uint32_t* d_mSubPopCount,
                            uint32_t* d_nSubPopSize,
                            uint32_t* d_mSubPopSize,
                            const uint32_t* d_subPopTotal)
{
    // Initialize the subpopulation index.
    uint32_t subPop = blockIdx.x*blockDim.x + threadIdx.x;

    // If sub-population index is well defined, count individuals over uniform distribution.
    if (subPop < d_subPopTotal[0])
    {
        d_nSubPopCount[subPop] = 0;
        d_mSubPopCount[subPop] = 0;
        d_nSubPopSize[subPop] = 0;
        d_mSubPopSize[subPop] = 0;
    }
}

// CUDA kernel which initializes the subpopulation all individuals belong to.
__global__ void subPopInitial(subPopulation* d_subPopulation,
                               curandState_t* d_randStates,
                               uint32_t* d_subPopCount,
                               uint32_t* d_subPopSize,
                               const float* d_cumulativeComSize,
                               const uint32_t* d_size,
                               const uint32_t* d_subPopTotal,
                               const ArchitectureParam h_archParam)
{
    // Determine the individuals census index.
    uint32_t censusIndex = blockIdx.x*blockDim.x + threadIdx.x;

    // Get the active thread identity of the current thread.
    uint32_t activeThreadId = smId()*h_archParam.warpsPerSM*h_archParam.threadsPerWarp
        + warpId()*h_archParam.threadsPerWarp + laneId();

    // For all individuals defined in the census, randomly assign a subpopulation and
    // increment the subpopulation counter (variable) and size (fixed).
    if (censusIndex < __ldg(&d_size[0]))
    {
        // Generate a random number between 0 and 1.
        float subPopProb = curand_uniform(&d_randStates[activeThreadId]);

        // Initialize the subpopulation the individual is part of.
        subPopulation local_subPopulation = 0;

        // Determine the subpopulation of the individual by scanning through
        // the normalized cumulative community size array and assigning 
        // the community where the random probability stops exceeding
        // the cumulative community size.
        while (subPopProb > d_cumulativeComSize[local_subPopulation])
        {
            ++local_subPopulation;
        }

        // Record the subpopulation to global memory.
        atomicAdd(&d_subPopSize[local_subPopulation], 1);
        atomicAdd(&d_subPopCount[local_subPopulation], 1);
        d_subPopulation[censusIndex] = local_subPopulation;
    }
}

// "Inefficient" CUDA kernel which cumulatively sums the number of individuals assigned
// to each subpopulation, to give the locations of where census indices of the individuals
// of that subpopulation begin in the subpopulation array. Note that
// the first element of the index array is zero.
__global__ void subPopCumulativeSum(uint32_t* d_nSubPopLoc,
                                    uint32_t* d_mSubPopLoc,
                                    const uint32_t* d_nSubPopSize,
                                    const uint32_t* d_mSubPopSize,
                                    const uint32_t* d_subPopTotal)
{
    // Declare sub-population index.
    uint32_t subPop;

    // Initialize the position of subpopulation zero.
    d_nSubPopLoc[0] = 0;
    d_mSubPopLoc[0] = 0;

    // Compute the position for accessing each subpopulations members.
    for (subPop = 1; subPop < d_subPopTotal[0]; ++subPop)
    {
        d_nSubPopLoc[subPop] = d_nSubPopLoc[subPop - 1] + d_nSubPopSize[subPop - 1];
        d_mSubPopLoc[subPop] = d_mSubPopLoc[subPop - 1] + d_mSubPopSize[subPop - 1];
    }
}

// CUDA kernel which takes the census index of an individual and records it in the appropriate
// subpopulation position in the array which contains census indices for individuals for
// every subpopulation combination.
__global__ void subPopIndexing(uint32_t* d_subPopIndex,
                               uint32_t* d_subPopLoc,
                               const subPopulation* d_subPopulation,
                               const uint32_t* d_reserveSize)
{
    // Determine the individuals census index.
    uint32_t censusIndex = blockIdx.x*blockDim.x + threadIdx.x;

    // If the census index is valid, find the location of the sub-population ordered
    // census index array for the individuals sub-population.
    if (censusIndex < d_reserveSize[0])
    {
        uint32_t locationIndex = atomicAdd(&d_subPopLoc[d_subPopulation[censusIndex]], 1);
        d_subPopIndex[locationIndex] = censusIndex;
    }
}

// Host function which calls the CUDA kernels for the cumulative survival functions, and initializes
// the human and mosquito populations, and creates the sub-population ordered array of population
// indices used during infection.
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
                                        const uint32_t h_mSize,
                                        const uint32_t h_subPopTotal,
                                        const Parameter h_parameter)
{
    // Call the CUDA kernel which initializes the subpopulation counters for the human and
    // mosquito populations.
    uint32_t subPopGridSize = static_cast<uint32_t>(ceil(h_subPopTotal / static_cast<float>(C_THREADSPERBLOCK)));
    subPopCount<<< subPopGridSize, C_THREADSPERBLOCK >>>
        (d_nSubPopCount, d_mSubPopCount, d_nSubPopSize, d_mSubPopSize, d_subPopTotal);

    // Initialize the number of blocks (gridSize) required on the GPU given the number of threads desired to be used on
    // each block. Need as many threads as individuals.
    uint32_t nGridSize = static_cast<uint32_t>(ceil(h_nSize / static_cast<float>(C_THREADSPERBLOCK)));
    uint32_t mGridSize = static_cast<uint32_t>(ceil(h_mSize / static_cast<float>(C_THREADSPERBLOCK)));

    // Call the CUDA kernels for initialization of human and mosquito subpopulation assignment, and increment
    // the counters for the number of individuals of each subpopulation.
    subPopInitial<<< nGridSize, C_THREADSPERBLOCK >>>
        (d_nSubPopulation, d_randStates, d_nSubPopCount, d_nSubPopSize, d_nCumulativeComSize, d_nSize, d_subPopTotal, h_parameter.arch);
    subPopInitial<<< mGridSize, C_THREADSPERBLOCK >>>
        (d_mSubPopulation, d_randStates, d_mSubPopCount, d_mSubPopSize, d_mCumulativeComSize, d_mSize, d_subPopTotal, h_parameter.arch);

    // Call the CUDA kernel which contains the indices of where each subpopulation begins
    // in the sub-population ordered census index array.
    subPopCumulativeSum<<< 1, 1 >>>(d_nSubPopLoc, d_mSubPopLoc, d_nSubPopSize, d_mSubPopSize, d_subPopTotal);

    // Use the array of subpopulation locations to record the census index of individuals 
    // into the sub-population ordered census index arary.
    subPopIndexing<<< nGridSize, C_THREADSPERBLOCK >>>
        (d_nSubPopIndex, d_nSubPopLoc, d_nSubPopulation, d_nSize);
    subPopIndexing<<< mGridSize, C_THREADSPERBLOCK >>>
        (d_mSubPopIndex, d_mSubPopLoc, d_mSubPopulation, d_mSize);

    // Call the CUDA kernel again which contains the indices of where each subpopulation begins
    // in the sub-population ordered census index array (overrode in the previous kernel calls).
    subPopCumulativeSum<<< 1, 1 >>>(d_nSubPopLoc, d_mSubPopLoc, d_nSubPopSize, d_mSubPopSize, d_subPopTotal);

    // Initialize the cumulative survival functions for human and mosquito life expectancy.
    nCumulativeSurvival<<< 1, 1 >>>(d_nSurvival, h_parameter.simulation);
    mCumulativeSurvival<<< 1, 1 >>>(d_mSurvival, h_parameter.simulation);

    // Call the CUDA kernels for initialization of the human and mosquito populations.
    nDemographicInitialization<<< nGridSize, C_THREADSPERBLOCK >>>
        (d_nAge, d_nHistory, d_nInfectStatus, d_nPLE, d_randStates, d_nSurvival, d_nSize, h_parameter);
    mDemographicInitialization<<< mGridSize, C_THREADSPERBLOCK >>>
        (d_mAge, d_mDead, d_mInfectStatus, d_mPLE, d_mSubPopulation, d_randStates, d_mDeadCount, d_mSubPopCount, d_mSurvival, d_mExpectedPopSize, d_mSize, h_parameter);
}