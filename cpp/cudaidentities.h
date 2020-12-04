// Header file containing CUDA intrinsics for determining the warp ID,
// SM ID (streaming multi-processor), and lane ID of a thread. This is used
// to compute the active thread index for a thread.
    
#include "host_defines.h"       // Definition of device-only functions.
#include <cstdint>              // Fixed-width integers.

// Start of the header guard.
#ifndef CUDAIDENTITIES_H
#define CUDAIDENTITIES_H

// Device function outputting the warp ID of a thread. This is the
// identity of the warp on each streaming multi-processor.
static __device__ __inline__ uint32_t warpId()
{
    uint32_t warpid;
    asm("mov.u32 %0, %warpid;" : "=r"(warpid));
    return warpid;
}

// Device function outputting the stream-multiprocessor ID of a thread.
// There are typically eight streaming-multiprocessors.
static __device__ __inline__ uint32_t smId()
{
    uint32_t smid;
    asm("mov.u32 %0, %smid;" : "=r"(smid));
    return smid;
}

// Device function outputting the lane ID of a thread. This is the
// thread ID of the warp.
static __device__ __inline__ uint32_t laneId()
{
    uint32_t laneid;
    asm("mov.u32 %0, %laneid;" : "=r"(laneid));
    return laneid;
}

#endif