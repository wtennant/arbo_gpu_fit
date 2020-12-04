// This header file contains the class for GPU architecture properties.
#include <cstdint>      // Fixed-width integers.

// Start of the header guard.
#ifndef ARCHITECTURECLASS_H
#define ARCHITECTURECLASS_H

class ArchitectureParam
{
public:
    uint32_t threadsPerWarp;        // Number of threads per warp.
    uint32_t warpsPerSM;            // Number of warps per Streaming Multiprocessor (SM).
    uint32_t totalSM;               // Total number of SMs.
    ArchitectureParam();            // Default constructor.
};

#endif