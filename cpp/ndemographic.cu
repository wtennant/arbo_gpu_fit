// ndemographics.cu: This file contains the host code which invokes CUDA kernels 
// for the simulation of the humans demographics.

#include "censustypedef.h"              // Type definitions for census data.
#include "constant.h"                   // Constants for simulation.
#include "curand_kernel.h"              // Required for random number generation in CUDA.
#include "cudaidentities.h"             // Streaming multiprocessor ID, warp ID, and lane ID.
#include "device_launch_parameters.h"   // Thread ID and block ID.
#include "Parameter.h"                  // Parameter space definition.

// Gives the cumulative bi-Weibull human life expectancy distribution probability. This disitrubion
// gives the most flexibility in age-specific mortality rates. 
__device__ __forceinline__ float device_nLifeExpectancy(const age local_age, const SimulationParam h_simParam)
{
    // Initialize as -(x / scale)^shape
    float dLifeExpectancy = powf(local_age * static_cast<float>(h_simParam.nScaleInfantMortality), h_simParam.nShapeInfantMortality);

    // Need more terms if passed the location where the second (burn out) Weibull distribution begins
    if (local_age < h_simParam.nLocWeibull)
    {
        return (1.0f - expf(-dLifeExpectancy));
    }
    else
    {
        dLifeExpectancy = dLifeExpectancy +
            powf((local_age - h_simParam.nLocWeibull) / h_simParam.nScaleLifeExpectancy, h_simParam.nShapeLifeExpectancy);
        return (1.0f - expf(-dLifeExpectancy));
    }
}

// CUDA kernel determining if an individual dies at the current time step.
__global__ void nBlockBirthDeath(age* d_nAge,
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
                                 const uint32_t* d_nSize,
                                 const uint32_t* d_subPopTotal,
                                 const Parameter h_parameter)
{
    // Determine which human belongs to each thread.
    uint32_t nIndex = threadIdx.x + blockDim.x * blockIdx.x;

    // Read in the number of humans in the population.
    uint32_t local_nSize = __ldg(&d_nSize[0]);

    // If the human belonging to the thread is valid.
    if (nIndex < local_nSize)
    {
        // Load in the individuals age and if they are susceptible, infected, or infectious.
        age local_age = (++d_nAge[nIndex]);
        infectStatus local_infectStatus = d_nInfectStatus[nIndex];

        // Determine if the human dies from the cumulative life expectancy distribution and the 
        // individuals pre-determined life expectancy probability.
        if (__ldg(&d_nPLE[nIndex]) > device_nLifeExpectancy(local_age, h_parameter.simulation))
        {
            // Check if an infected, but not infectious individual is due to become infectious, and
            // check if an infectious individual has reached age of recovery from the disease.
            if ((local_infectStatus == 2) && (d_nRecovery[nIndex] < local_age))
            {
                atomicSub(&d_nInfectedCount[d_nStrain[nIndex] * __ldg(&d_subPopTotal[0]) + __ldg(&d_nSubPopulation[nIndex])], 1);
                d_nInfectStatus[nIndex] = 0;
            }
            else if ((local_infectStatus == 1) && (d_nExposed[nIndex] < local_age))
            {
				atomicAdd(&d_nIncidenceSubPop[d_nStrain[nIndex] * __ldg(&d_subPopTotal[0]) + __ldg(&d_nSubPopulation[nIndex])], 1);
                atomicAdd(&d_nInfectedCount[d_nStrain[nIndex] * __ldg(&d_subPopTotal[0]) + __ldg(&d_nSubPopulation[nIndex])], 1);
                d_nInfectStatus[nIndex] = 2;
            }
        }
        else
        {
            // If the human was infected with a virus when they died, decrease counter containing the total 
            // number of infected individuals for the subpopulation they belong to, for that strain.
            if (local_infectStatus == 2)
            {
                atomicSub(&d_nInfectedCount[d_nStrain[nIndex] * __ldg(&d_subPopTotal[0]) + +__ldg(&d_nSubPopulation[nIndex])], 1);
            }

            // Initialize the active thread ID used for random number generation.
            uint32_t activeThreadId = smId()*h_parameter.arch.warpsPerSM*h_parameter.arch.threadsPerWarp
                + warpId()*h_parameter.arch.threadsPerWarp + laneId();

            // Create a new human in place of the human that died.
            d_nAge[nIndex] = 0;
            d_nInfectStatus[nIndex] = 0;
            d_nPLE[nIndex] = curand_uniform(&d_randStates[activeThreadId]);

            // Reset the humans immunological history.
            strain local_strain{ 0 };
            while (local_strain < C_STRAINS)
            {
                d_nHistory[local_strain*local_nSize + nIndex] = -1;
                ++local_strain;
            }
        }
    }
}

// Reset the daily incidence of each strain in each sub-population to zero.
__global__ void resetIncidence(uint32_t* d_nIncidenceSubPop,
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

// The CPU code which calls all CUDA kernels involved in the update of human demographics.
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
                           const Parameter h_parameter)
{
	// Determine the size of the grid of blocks required on the GPU to reset daily incidence.
	uint32_t resetGridSize = static_cast<uint32_t>(ceil(h_parameter.simulation.metaPop * C_STRAINS / static_cast<float>(C_THREADSPERBLOCK)));

	// Reset the daily incidence counter fo each strain in each sub-population to zero, ready
	// to count each new case for that day.
	resetIncidence<<< resetGridSize, C_THREADSPERBLOCK >>>(d_nIncidenceSubPop, d_subPopTotal);

    // Determine the size of the grid of blocks required on the GPU to do the update of human demographics.
    uint32_t gridSize = static_cast<uint32_t>(ceil(h_nSize / static_cast<float>(C_THREADSPERBLOCK)));

    // Call the kernel which determines if a human dies or not (and is replaced if they do die).
    nBlockBirthDeath<<< gridSize, C_THREADSPERBLOCK >>>
        (d_nAge, d_nExposed, d_nHistory, d_nInfectStatus, d_nPLE, d_nRecovery, d_nStrain, d_nSubPopulation, 
        d_randStates, d_nInfectedCount, d_nIncidenceSubPop, d_nSize, d_subPopTotal, h_parameter);
}