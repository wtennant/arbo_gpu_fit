// ArchitectureParam.cpp: Default constructor for GPU architectural properties.
// This implementation saves on relying on user input/knowledge
// of their own architecture and offers better portability between different GPUs.

#include "cuda_runtime_api.h"   // CUDA functions for getting GPU properties.
#include "ArchitectureParam.h"  // GPU architectural properties definition.

// Gets the architectural properties of the GPU. 
ArchitectureParam::ArchitectureParam()
{
    // CUDA device properties can only write to an int.
    int value;
    
    // Get the number of threads in a warp.
    cudaDeviceGetAttribute(&value, cudaDevAttrWarpSize, 0);
    threadsPerWarp = value;

    // Get the number of Streaming Multiprocessors (SMs) on the device.
    cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, 0);
    totalSM = value;

    // Get the total number of threads per SM, then calculate the number of warps per SM.
    cudaDeviceGetAttribute(&value, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
    warpsPerSM = value / threadsPerWarp;
}