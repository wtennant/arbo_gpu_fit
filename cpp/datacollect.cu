// datacollect.cu: contains the host code which invokes CUDA kernels for summing up 
// the number of infected/total individuals across all sub-populations, and kernel code
// for storing the result in time series data.

#include "device_launch_parameters.h"   // CUDA thread and block index.
#include "reduction.cuh"                // Reduction/summation CUDA kernel.
#include "constant.h"                   // Number of threads per block in summation.

// CUDA kernel for copying infected data at the subpopulation level into the time series device memory.
__global__ void saveInfectedSubPop(uint32_t* d_nInfectedSubPopCountSeries,
                                   const uint32_t* d_nInfectedSubPopCount,
                                   const uint32_t h_subPopTotal,
                                   const uint32_t t)
{
    // Initialize the (subPop, strain) combination the thread is resposible for.
    uint32_t subPopStrain = threadIdx.x + blockIdx.x*blockDim.x;

    // Check that the subPop, strain combination is valid is valid.
    if (subPopStrain < C_STRAINS*h_subPopTotal)
    {
        // Copy the total number of infected individuals in each subpopulation across to the time series data.
        d_nInfectedSubPopCountSeries[t*C_STRAINS*h_subPopTotal + subPopStrain] = d_nInfectedSubPopCount[subPopStrain];
    }
}

// CUDA kernel for copying reduced infected data into the time series device memory.
__global__ void saveInfected(uint32_t* d_nInfectedCount,
                             const uint32_t* d_nReductionInfectedCount,
                             const uint32_t t)
{
    // Initialize the strain the thread is resposible for.
    uint32_t strain = threadIdx.x + blockIdx.x*blockDim.x;

    // Check that the strain is valid.
    if (strain < C_STRAINS)
    {
        // Copy the total number of infected individuals across to the time series data.
        d_nInfectedCount[t*C_STRAINS + strain] = d_nReductionInfectedCount[strain];
    }
}

// CUDA kernel for copying a specific sub-populations infected data into the
// time series device memory. This will be used in determining local serotype
// co-ciculation levels.
__global__ void saveSpecialInfected(uint32_t* d_nOneSubPopInfectedCount,
                                    const uint32_t* d_nInfectedSubPopCount,
                                    const uint32_t h_specialSubPop,
                                    const uint32_t h_subPopTotal,
                                    const uint32_t t)
{
    // Initialize the strain the thread is resposible for.
    uint32_t strain = threadIdx.x + blockIdx.x*blockDim.x;

    // Check that the strain is valid.
    if (strain < C_STRAINS)
    {
        // Copy the total number of infected individuals across to the time series data.
        d_nOneSubPopInfectedCount[t*C_STRAINS + strain] =
            d_nInfectedSubPopCount[strain*h_subPopTotal + h_specialSubPop];
    }
}

// CUDA kernel for copying reduced mosquito and human population size data into 
// the time series device memory.
__global__ void saveCount(uint32_t* d_nCount,
                          uint32_t* d_mCount,
                          const uint32_t* d_nReductionCount,
                          const uint32_t* d_mReductionCount,
                          const uint32_t t)
{
    // Copy the total number of human and mosquito individuals across to the time series data.
    d_nCount[t] = d_nReductionCount[0];
    d_mCount[t] = d_mReductionCount[0];
}

// Host code which invokes CUDA kernels for summing up the number of infected/total 
// individuals across all sub-populations, and kernel code for storing the result 
// in time series data.
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
                          const uint32_t t)
{
    // Compute the number of blocks required to sum the total number of individuals 
    // across all sub-populations.
    uint32_t reductionSize{static_cast<uint32_t>(ceil(h_subPopTotal / static_cast<float>(C_THREADSPERBLOCKSUM)))};
    reductionSize = static_cast<uint32_t>(ceil(reductionSize / 2.0f));
    blockSingleSum<C_THREADSPERBLOCKSUM, uint32_t>
        <<< reductionSize, C_THREADSPERBLOCKSUM, sizeof(uint32_t)*C_THREADSPERBLOCKSUM >>>
        (d_nReductionCount, d_nSubPopCount, h_subPopTotal, reductionSize);
    blockSingleSum<C_THREADSPERBLOCKSUM, uint32_t>
        <<< reductionSize, C_THREADSPERBLOCKSUM, sizeof(uint32_t)*C_THREADSPERBLOCKSUM >>>
        (d_mReductionCount, d_mSubPopCount, h_subPopTotal, reductionSize);

    // The sum kernel can only sum at most 2*C_THREADSPERBLOCKSUM
    // old blocks together at a time. Keep calling the kernel, until all old blocks 
    // (oldReductionSize) can be summed on one new block (reductionSize).
    while (reductionSize > 1)
    {
        uint32_t oldReductionSize = reductionSize;
        reductionSize = static_cast<uint32_t>(ceil(reductionSize / static_cast<float>(C_THREADSPERBLOCKSUM)));
        reductionSize = static_cast<uint32_t>(ceil(reductionSize / 2.0f));
        blockSingleSum<C_THREADSPERBLOCKSUM, uint32_t>
            <<< reductionSize, C_THREADSPERBLOCKSUM, sizeof(uint32_t)*C_THREADSPERBLOCKSUM >>>
            (d_nReductionCount, d_nReductionCount, oldReductionSize, reductionSize);
        blockSingleSum<C_THREADSPERBLOCKSUM, uint32_t>
            <<< reductionSize, C_THREADSPERBLOCKSUM, sizeof(uint32_t)*C_THREADSPERBLOCKSUM >>>
            (d_mReductionCount, d_mReductionCount, oldReductionSize, reductionSize);
    }

    // Compute the number of blocks required to sum the total number of infected individuals across all sub-populations.
    reductionSize = static_cast<uint32_t>(ceil(h_subPopTotal / static_cast<float>(C_THREADSPERBLOCKSUM)));
    reductionSize = static_cast<uint32_t>(ceil(reductionSize / 2.0f));
    uint32_t infectedSize{ reductionSize*C_STRAINS };
    blockSingleSum<C_THREADSPERBLOCKSUM, uint32_t>
        <<< infectedSize, C_THREADSPERBLOCKSUM, sizeof(uint32_t)*C_THREADSPERBLOCKSUM >>>
        (d_nReductionInfectedCount, d_nInfectedSubPopCount, h_subPopTotal, reductionSize);

    // The sum kernel can only sum at most 2*C_THREADSPERBLOCKSUM
    // old blocks together at a time. Keep calling the kernel, until all old blocks 
    // (oldReductionSize) can be summed on one new block (reductionSize).
    while (reductionSize > 1)
    {
        uint32_t oldReductionSize = reductionSize;
        reductionSize = static_cast<uint32_t>(ceil(reductionSize / static_cast<float>(C_THREADSPERBLOCKSUM)));
        reductionSize = static_cast<uint32_t>(ceil(reductionSize / 2.0f));
        infectedSize = reductionSize*C_STRAINS;
        blockSingleSum<C_THREADSPERBLOCKSUM, uint32_t>
            <<< infectedSize, C_THREADSPERBLOCKSUM, sizeof(uint32_t)*C_THREADSPERBLOCKSUM >>>
            (d_nReductionInfectedCount, d_nReductionInfectedCount, oldReductionSize, reductionSize);
    }

	// Copy the sub-population infected data into time series memory on the device. 
	uint32_t infectedSubPopGridSize = static_cast<uint32_t>(ceil((C_STRAINS*h_subPopTotal) / static_cast<float>(C_THREADSPERBLOCKSUM)));
	saveInfectedSubPop << < infectedSubPopGridSize, C_THREADSPERBLOCKSUM >> >
		(d_nInfectedSubPopCountSeries, d_nInfectedSubPopCount, h_subPopTotal, t);

	// Copy the reduced infected data into time series memory on the device. 
	uint32_t infectedGridSize = static_cast<uint32_t>(ceil(C_STRAINS / static_cast<float>(C_THREADSPERBLOCKSUM)));
	saveInfected << < infectedGridSize, C_THREADSPERBLOCKSUM >> >
		(d_nInfectedCount, d_nReductionInfectedCount, t);

	// Compute the number of blocks required to sum the total number of cases across all sub-populations.
	reductionSize = static_cast<uint32_t>(ceil(h_subPopTotal / static_cast<float>(C_THREADSPERBLOCKSUM)));
	reductionSize = static_cast<uint32_t>(ceil(reductionSize / 2.0f));
	infectedSize = reductionSize * C_STRAINS;
	blockSingleSum<C_THREADSPERBLOCKSUM, uint32_t>
		<<< infectedSize, C_THREADSPERBLOCKSUM, sizeof(uint32_t)*C_THREADSPERBLOCKSUM >>>
		(d_nReductionInfectedCount, d_nIncidenceSubPop, h_subPopTotal, reductionSize);

	// The sum kernel can only sum at most 2*C_THREADSPERBLOCKSUM
	// old blocks together at a time. Keep calling the kernel, until all old blocks 
	// (oldReductionSize) can be summed on one new block (reductionSize).
	while (reductionSize > 1)
	{
		uint32_t oldReductionSize = reductionSize;
		reductionSize = static_cast<uint32_t>(ceil(reductionSize / static_cast<float>(C_THREADSPERBLOCKSUM)));
		reductionSize = static_cast<uint32_t>(ceil(reductionSize / 2.0f));
		infectedSize = reductionSize * C_STRAINS;
		blockSingleSum<C_THREADSPERBLOCKSUM, uint32_t>
			<<< infectedSize, C_THREADSPERBLOCKSUM, sizeof(uint32_t)*C_THREADSPERBLOCKSUM >>>
			(d_nReductionInfectedCount, d_nReductionInfectedCount, oldReductionSize, reductionSize);
	}

	// Copy the reduced incidence data into time series memory on the device. 
	infectedGridSize = static_cast<uint32_t>(ceil(C_STRAINS / static_cast<float>(C_THREADSPERBLOCKSUM)));
	saveInfected<<< infectedGridSize, C_THREADSPERBLOCKSUM >>>
		(d_nIncidence, d_nReductionInfectedCount, t);
	
    // Copy the single subpopulations infected data into time series memory on the device. 
    uint32_t specialInfectedGridSize = static_cast<uint32_t>(ceil(C_STRAINS / static_cast<float>(C_THREADSPERBLOCKSUM)));
    saveSpecialInfected<<< specialInfectedGridSize, C_THREADSPERBLOCKSUM >>>
        (d_nOneSubPopInfectedCount, d_nInfectedSubPopCount, h_specialSubPop, h_subPopTotal, t);

    // Copy the reduced population counts for human and mosquitoes into time series device memory.
    saveCount<<< 1, 1 >>>(d_nCount, d_mCount, d_nReductionCount, d_mReductionCount, t);
}