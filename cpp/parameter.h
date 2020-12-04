// Parameter.h: contains the class for parameters for the dengue simulation.

#include "ArchitectureParam.h"  // GPU architecture parameters.
#include "IOParam.h"	        // Input-output parameters.
#include "SimulationParam.h"    // Simulation parameters.

// Start of the header guard.
#ifndef PARAMETER_H
#define PARAMETER_H

class Parameter
{
public:
    ArchitectureParam arch;                         // GPU architecture parameters.
    IOParam io;                                     // Input-output parameters.
    SimulationParam simulation;                     // Simulation parameters.
    Parameter() : arch(), io(), simulation() {};    // Default constructor.
    void writeParameters(uint32_t h_mcmcStep);
};

#endif