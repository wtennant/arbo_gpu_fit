// nepidemic.cu: CUDA kernels for determining the number of transmission events that should occur
// to the individuals of each subpopulation and infecting humans.

#include "curand_kernel.h"              // Required for random number generation in CUDA.
#include "cudaidentities.h"             // Streaming multiprocessor ID, warp ID, and lane ID.
#include "device_launch_parameters.h"   // Thread ID and block ID.
#include "censustypedef.h"              // Type definitions for census data.
#include "Parameter.h"                  // Parameter space definition.
#include "constant.h"                   // Constants for simulation.

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

// CUDA kernel which determines the number of transmission events that should occur to
// individuals in every subpopulation.
__global__ void nVisitingInfected(curandState_t* d_randStates,
                                  uint32_t* d_nTransmission,
                                  const uint32_t* d_infectedCount,
                                  const uint32_t* d_nSubPopCount,
                                  const uint32_t* d_mSubPopCount,
                                  const float* d_sparseNetWeight,
                                  const uint32_t* d_sparseNetTo,
                                  const uint32_t* d_sparseNetLoc,
                                  const uint32_t* d_subPopTotal,
                                  const Parameter h_parameter)
{
    // Initialize the strain, subpopulation combination of the thread.
    uint32_t strainSubPop = blockIdx.x*blockDim.x + threadIdx.x;

    // Load in the total number of subpopulation from global memory to L1 cache.
    uint32_t subPopTotal = __ldg(&d_subPopTotal[0]);

    // Check if the strain, subpopulation combination is within limits.
    if (strainSubPop < subPopTotal*C_STRAINS)
    {
        // Load the number of infected mosquitoes in that subpopulation, strain combination.
        uint32_t infectedCount = __ldg(&d_infectedCount[strainSubPop]);

        // If there are infected individuals, then compute the number of transmission events
        // that they would cause.
        if (infectedCount > 0)
        {
            // Determine the source subpopulation of the thread.
            uint32_t fromSubPop = strainSubPop % subPopTotal;

            // Initialize the active thread index used for random number generation.
            uint32_t activeThreadId = smId()*h_parameter.arch.warpsPerSM*h_parameter.arch.threadsPerWarp
                + warpId()*h_parameter.arch.threadsPerWarp + laneId();

            // Load in the random number generator into local memory.
            curandState_t local_state = d_randStates[activeThreadId];

            // Initialize the ratio between the number of mosquitoes and the number of humans.
            float mosquitoHumanRatio = d_mSubPopCount[fromSubPop] / static_cast<float>(d_nSubPopCount[fromSubPop]);

            // Determine the total number of transmission events that the humans would cause to mosquitoes.
            uint32_t transmissions = decimalResolve(&local_state,
                static_cast<float>(mosquitoHumanRatio*h_parameter.simulation.bitingRate*h_parameter.simulation.nmBitingSuccess*infectedCount));

            // Delcare a variable for how many transmission events are not dispersed to a different community.
            uint32_t noDistanceTransmissions;

            // Initialize the location in global memory where transmission numbers to individuals
            // of every subpopulation begins for each strain.
            uint32_t strainStartIndex = subPopTotal*(strainSubPop / subPopTotal);

            // Get the location in the network weight array where the source subpopulations
            // non-zero weights begin. 
            uint32_t sparseNetLoc = d_sparseNetLoc[fromSubPop];

            // Get the number of connections that the source subpopulation has.
            uint32_t connections = d_sparseNetLoc[fromSubPop + 1] - sparseNetLoc;

            // Only send transmission events to other communities if the source community is not isolated.
            if (connections > 0)
            {
                // Compute the number of transmission events that remain in the source community.
                noDistanceTransmissions = decimalResolve(&local_state, transmissions*h_parameter.simulation.noDistance);
                transmissions -= noDistanceTransmissions;

                // Calculate the total number of long-distance transmissions to random communities.
                uint32_t forcedLongDistance = decimalResolve(&local_state, transmissions*h_parameter.simulation.longDistance);
                transmissions -= forcedLongDistance;

                // Declare the subpopulation that each transmission event is sent to.
                uint32_t toSubPop;

                // Provided there are still local transmission numbers to be made.
                while (transmissions > 0)
                {
                    // Generate a random number between [0 and connections - 1].
                    toSubPop = static_cast<uint32_t>(curand(&local_state) % connections);

                    // In the current revision of the code, we know the weight is 1, so
                    // just get the destination subpopulation, or column index, in the sparse matrix.
                    toSubPop = d_sparseNetTo[sparseNetLoc + toSubPop];

                    // Increment the number of transmissions that should be placed upon individuals
                    // of the destination subpopulation.
                    atomicAdd(&d_nTransmission[strainStartIndex + toSubPop], 1);

                    // Decrease the number of local transmission events to disperse.
                    --transmissions;
                }

                // Provided there are long distance transmission events to make, randomly choose
                // a subpopulation within the lattice, and disperse the transmission event to the
                // individuals of that subpopulation.
                while (forcedLongDistance > 0)
                {
                    toSubPop = static_cast<uint32_t>(curand(&local_state) % subPopTotal);
                    atomicAdd(&d_nTransmission[strainStartIndex + toSubPop], 1);
                    --forcedLongDistance;
                }
            }
            else
            {
                // If the source community has no neighbours, all transmission events remain within that
                // community.
                noDistanceTransmissions = transmissions;
            }

            // Add on the number of non-dispersed transmission events to the source community.
            atomicAdd(&d_nTransmission[strainStartIndex + fromSubPop], noDistanceTransmissions);

            // Store the random number generator state back to global memory.
            d_randStates[activeThreadId] = local_state;
        }
    }
}

// CUDA kernel which infects the humans of every subpopulation given the number
// of transmission events that should occur to humans in that subpopulation.
__global__ void mnTransmission(age* d_nAge,
                               exposed* d_nExposed,
                               history* d_nHistory,
                               infectStatus* d_nInfectStatus,
                               recovery* d_nRecovery,
                               strain* d_nStrain,
                               curandState_t* d_randStates,
                               uint32_t* d_nAgeOfInfection,
                               uint32_t* d_nAgeOfInfectionCount,
                               const uint32_t* d_nInfectedCount,
                               const uint32_t* d_nSubPopIndex,
                               const uint32_t* d_nSubPopLoc,
                               const uint32_t* d_nSubPopSize,
                               const uint32_t* d_mnTransmission,
                               const uint32_t* d_nSize,
                               const uint32_t* d_subPopTotal,
                               const uint32_t t,
                               const uint32_t maxTime,
                               const Parameter h_parameter)
{
    // Initialize the subpopulation of the thread.
    uint32_t subPop = blockIdx.x*blockDim.x + threadIdx.x;

    // Load in the total number of subpopulation from global memory to L1 cache.
    uint32_t subPopTotal = __ldg(&d_subPopTotal[0]);

    // Initialize the active thread index used for random number generation.
    uint32_t activeThreadId = smId()*h_parameter.arch.warpsPerSM*h_parameter.arch.threadsPerWarp
        + warpId()*h_parameter.arch.threadsPerWarp + laneId();

    // Check if the subpopulation of the thread is within limits.
    if (subPop < subPopTotal)
    {
        // Load in the maximum number of individuals in the subpopulation.
        uint32_t subPopSize = __ldg(&d_nSubPopSize[subPop]);

        // Check that individuals exist in the subpopulation.
        if (subPopSize > 0)
        {
            // Read in the random number generator state into global memory.
            curandState_t local_state = d_randStates[activeThreadId];

            // Load in the location where to begin in the sub-population ordered census
            // indices for the human population for the subpopulation of the thread.
            uint32_t subPopLoc = __ldg(&d_nSubPopLoc[subPop]);

            // Load in the total number of humans alive in the subpopulation.
            uint32_t local_nSize = __ldg(&d_nSize[0]);

            // Randomly choose a strain of the virus to begin infecting individuals of the 
            // subpopulation with.
            uint32_t local_strain = curand(&local_state) % C_STRAINS;

            // Cycle through all the strains, infecting humans give the total number of transmission events that
            // occur to individuals of the subpopulation per strain.
            for (uint32_t strainCount = 0; strainCount < C_STRAINS; ++strainCount)
            {
                // Load in the number of transmission events.
                uint32_t transmissionCount = d_mnTransmission[local_strain*subPopTotal + subPop];

                // Provided there are transmission events:
                while (transmissionCount > 0)
                {
                    // Choose a random individual in the sub-population.
                    uint32_t censusIndex = curand(&local_state) % subPopSize;

                    // Find their index within the census data by using the sub-population ordered
                    // census indices.
                    censusIndex = d_nSubPopIndex[subPopLoc + censusIndex];

                    // If their census index is valid (which it should be by construction anyway),
                    // continue with transmission.
                    if (censusIndex < local_nSize)
                    {
                        // The virus is transmitted to the individual, so decrease the number of 
                        // remaining transmission events to make.
                        --transmissionCount;

                        // The individual will be infected if it is not already infected, or 
                        // if it is not immune to the strain of the virus.
                        if ((!d_nInfectStatus[censusIndex]) && (d_nHistory[local_strain*local_nSize + censusIndex] == -1))
                        {
                            // If the simulation is within the last 10 years, record the age of the individual being infected.
                            if (maxTime - t <= (10 * C_YEAR))
                            {
                                // Compute the number of previous infections the individual has.
                                uint32_t prevInfect = 0;
                                for (uint32_t s = 0; s < C_STRAINS; ++s)
                                {
                                    prevInfect += ((d_nHistory[s*local_nSize + censusIndex] == -1) ? 0 : 1);
                                }

                                // Increment the appropriate number of ages recorded for the first, second, third and fourth infection.
                                uint32_t infNumber = atomicInc(&d_nAgeOfInfectionCount[prevInfect], C_NAOIRECORD - 1);

                                // Save the age of exposure.
                                d_nAgeOfInfection[prevInfect*C_NAOIRECORD + infNumber] = d_nAge[censusIndex];
                            }

                            // Infect the individual.
                            d_nInfectStatus[censusIndex] = 1;                                              // Infected, not infectious.
                            d_nStrain[censusIndex] = local_strain;
                            d_nHistory[local_strain*local_nSize + censusIndex] =  d_nAge[censusIndex];     // Record age of exposure.

                            // Age at which individual becomes infectious.
                            d_nExposed[censusIndex] = static_cast<exposed>(decimalResolve(&local_state,
                                static_cast<float>(d_nAge[censusIndex] + h_parameter.simulation.nExposed)));

                            // Age at which individual stops being infected.
                            d_nRecovery[censusIndex] = static_cast<recovery>(decimalResolve(&local_state,
                                static_cast<float>(d_nExposed[censusIndex] + h_parameter.simulation.recovery)));
                        }
                    }
                }

                // Create a small chance of a transmission event occuring from external sources.
                // Introduction rate given as per day per strain, so need to convert to per day per strain per subpopulation.
                uint32_t extIntroduction = 0;
                extIntroduction = decimalResolve(&local_state,
                    static_cast<float>(h_parameter.simulation.externalInfection) / static_cast<float>(subPopTotal));

                // Define the number of attempts that have been made to externally introduce the virus.
                uint32_t introAttempt = 0;

                // Provided there is an introduction:
                while ((extIntroduction > 0) && (introAttempt < C_MAXINTROATTEMPT))
                {
                    // Increase the number of attempts that have been made to introduce the virus.
                    ++introAttempt;

                    // Choose a random individual in the sub-population.
                    uint32_t censusIndex = curand(&local_state) % subPopSize;

                    // Find their index within the census data by using the sub-population ordered
                    // census indices.
                    censusIndex = d_nSubPopIndex[subPopLoc + censusIndex];

                    // If their census index is valid (which it should be by construction anyway),
                    // continue with transmission.
                    if (censusIndex < local_nSize)
                    {
                        // The individual will be infected if it is not already infected, or 
                        // if it is not immune to the strain of the virus.
                        if ((!d_nInfectStatus[censusIndex]) && (d_nHistory[local_strain*local_nSize + censusIndex] == -1))
                        {
                            // The virus is transmitted to the individual, so decrease the number of 
                            // remaining introduction events to make.
                            --extIntroduction;

                            // Infect the individual.
                            d_nInfectStatus[censusIndex] = 1;                                               // Infected, not infectious.
                            d_nStrain[censusIndex] = local_strain;
                            d_nHistory[local_strain*local_nSize + censusIndex] = d_nAge[censusIndex];       // Record age of exposure.

                            // Age at which individual becomes infectious.
                            d_nExposed[censusIndex] = static_cast<exposed>(decimalResolve(&local_state,
                                static_cast<float>(d_nAge[censusIndex] + h_parameter.simulation.nExposed)));

                            // Age at which individual stops being infected.
                            d_nRecovery[censusIndex] = static_cast<recovery>(decimalResolve(&local_state,
                                static_cast<float>(d_nExposed[censusIndex] + h_parameter.simulation.recovery)));
                        }
                    }
                }

                // Move onto the transmission events of the next strain.
                local_strain = (local_strain + 1) % C_STRAINS;
            }

            // Store the random number generator back to global memory.
            d_randStates[activeThreadId] = local_state;
        }
    }
}