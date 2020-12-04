// setuprng.cu :: Initializes the random states for CUDA random number generation.
// There is one RNG for each possible active thread on the GPU.

#include <chrono>                       // System clock for seed generation.
#include "curand_kernel.h"              // CUDA random number generation.
#include "device_launch_parameters.h"   // Block ID and thread ID.
#include "ArchitectureParam.h"          // GPU architectural properties definition.
#include "constant.h"                   // Constants for simulation.

// CUDA kernel which initializes the random number generator states.
__global__ void setupRandStates(curandState_t* d_randStates,
                                unsigned int seed,
                                uint32_t totalActiveThreads)
{
    // Initialize the active thread index for the CUDA thread.
    uint32_t activeThreadId = blockIdx.x*blockDim.x + threadIdx.x;

    // Provided the active thread index is valid, initialize the 
    // random number generator state.
    if (activeThreadId < totalActiveThreads)
    {
        curand_init(seed, activeThreadId, 0, &d_randStates[activeThreadId]);
    }
}

// Host function which calls the kernel which initializes the CUDA random number generator states.
__host__ void setupCudaRNG(curandState_t* d_randStates, 
                           ArchitectureParam h_architecture)
{
    // Initialize random number generator seed from the system clock time.
    unsigned int seed{ static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()) };

    // Initialize the total possible number of active threads on the GPU.
    // Initialize the grid size to use in initialization of random number generators.
    uint32_t totalActiveThreads{ h_architecture.totalSM*h_architecture.warpsPerSM*h_architecture.threadsPerWarp };
    uint32_t setupRandGridSize{ static_cast<uint32_t>(ceil(totalActiveThreads / static_cast<float>(C_THREADSPERBLOCK))) };

    // Call the kernel which initializes the random number generator states.
    setupRandStates<<< setupRandGridSize, C_THREADSPERBLOCK >>>(d_randStates, seed, totalActiveThreads);
}