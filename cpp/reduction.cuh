// reduction.cuh: This CUDA header file contains the CUDA reduction kernels which compute the sum
// across input count arrays. For example, the input count array may countain counts per GPU block,
// yielding an output that sums the counts across all GPU blocks.

#include <cstdint>              // Fixed-width integers
#include "host_defines.h"       // Allows definition of CUDA kernels.

#ifndef REDUCTION_CUH
#define REDUCTION_CUH

// blockSingleSum is the CUDA kernel which sums across a vector of input count data. The kernel can sum at most
// twice the size of a GPU block number of elements at once. The kernel is called multiple times in cases where
// the number of elements exceeds twice the size of a GPU block. The kernel can also sum across multiple sets of count
// data concurrently provided the multiple sets are ordered within the input data (e.g. for each strain), in this case
// invoke the kernel over an integer multiple the reduction grid size.
template <uint32_t blockSize, typename T>
__global__ void blockSingleSum(T* d_outputCount, 
                               const T* d_inputCount,                               
                               const uint32_t oldReductionSize,
                               const uint32_t reductionSize)
{
    // Declare all the shared memory required on the GPU block. The amount of shared memory is dependent
    // on the number of threads in a block, thus the size of the shared memory is defined in the kernel call itself.
    extern __shared__ uint32_t sharingIsCaring[];
    uint32_t* countData = (uint32_t*)(&sharingIsCaring[0]);

    // Initialize the reduction index. This is the (local) index for one set of count data. 
    // reductionSize gives the number of blocks in the GPU grid required to reduce one set of count data.
    // Multiply by two here, since a single GPU block can sum up to two blocks worth of elements.
    uint16_t threadid = threadIdx.x;
    uint32_t redIndex = (blockIdx.x % reductionSize)*blockDim.x*2 + threadid;

    // Initialize the index for loading in from the input count data (of the previous reduction).
    uint32_t loadIndex = (blockIdx.x / reductionSize)*oldReductionSize + redIndex;

    // Initialize the shared variables to zero.
    countData[threadid] = 0;

    // If the reduction index is small enough such that a data element exists within that particular
    // set of input data, then load it in using the loading index. If the reduction index is small enough
    // such that the corresponding data element and a data element a block-width further in memory is within the same
    // set of input data, then load both in and "pre-"sum.
    if ((redIndex + blockDim.x) < oldReductionSize)
    {
        countData[threadid] = static_cast<uint32_t>(d_inputCount[loadIndex] + d_inputCount[loadIndex + blockDim.x]);
    }
    else if (redIndex < oldReductionSize)
    {
        countData[threadid] = static_cast<uint32_t>(d_inputCount[loadIndex]);
    }

    // Ensure all data has been loaded onto the shared variables before proceeding
    // with the summation/reduction.
    __syncthreads();

    // Depending on the size of each GPU block, add the elements half a block-width away until reaching only the
    // first 32 elements of the reduced count data are left to be summed (this is the same of a warp).
    if (blockSize >= 1024){ if (threadid < 512) { countData[threadid] += countData[threadid + 512]; } __syncthreads(); }
    if (blockSize >= 512){ if (threadid < 256) { countData[threadid] += countData[threadid + 256]; } __syncthreads(); }
    if (blockSize >= 256){ if (threadid < 128) { countData[threadid] += countData[threadid + 128]; } __syncthreads(); }
    if (blockSize >= 128){ if (threadid < 64) { countData[threadid] += countData[threadid + 64]; } __syncthreads(); }

    // Since the reduction is now occuring only on one warp, and instructions are issued per warp, the latter if
    // statements are no-longer required. If countData is declared as "volatile", the threadfence blocks are no longer
    // required either since operating on the shared data would force threads to store the result back.
    if (threadid < 32)
    {
        if (blockSize >= 64){ countData[threadid] += countData[threadid + 32]; __threadfence_block(); };
        if (blockSize >= 32){ countData[threadid] += countData[threadid + 16]; __threadfence_block(); };
        if (blockSize >= 16){ countData[threadid] += countData[threadid + 8]; __threadfence_block(); };
        if (blockSize >= 8){ countData[threadid] += countData[threadid + 4]; __threadfence_block(); };
        if (blockSize >= 4){ countData[threadid] += countData[threadid + 2]; __threadfence_block(); };
        if (blockSize >= 2){ countData[threadid] += countData[threadid + 1]; };
    }

    // Write the summed result back to global memory such that consecutive blocks store
    // in consecutive indices of the output count data.
    if (threadid == 0)
    {
        d_outputCount[blockIdx.x] = static_cast<T>(countData[0]);
    }
}

#endif