// Header file containing the declaration for the initialization of 
// different parameter sets for the dengue simulation.

#include <random>				// (Pseudo-)random number generation.
#include "parameterclass.h"     // Parameter space definition.

// Start of the header guard.
#ifndef PARAMETERFUNC_H
#define PARAMETERFUNC_H

// Parameter set initialization.
void initialParameter(Parameter* h_parameter,
                      uint32_t h_simRun,
                      std::mt19937 &h_rng);

// Function which parses the input command arguments.
void argParse(IOParam* ioParam,
			  int argc,
	          char* argv[]);

// Function which reads in a climate data file and and
// stores the required data in the climate parameter class.
void readClimate(ClimParam* climParam, 
                 const IOParam ioParam);

// Function that adjusts the parameters of the dengue simulation
// at each time step given climate data.
void changeParam(Parameter* h_parameter,
                 const ClimParam climParam,
                 const uint32_t t);

#endif