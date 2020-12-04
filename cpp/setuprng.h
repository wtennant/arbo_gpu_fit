// Header file containing the declarations for setting up random number
// generators on the GPU.

#include "curand_kernel.h"      // CUDA random number generation.
#include "ArchitectureParam.h"  // GPU architectural properties definition. 

#ifndef SETUPRNG_H
#define SETUPRNG_H

// Initializes the random number generators on the GPU.
__host__ void setupCudaRNG(curandState_t* d_randStates,
                           ArchitectureParam h_architecture);

#endif