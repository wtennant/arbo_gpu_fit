// Header file for network.cpp.

#include <cstdint>				// Fixed-width integers.
#include "igraph.h"				// Network library.
#include "Parameter.h"		    // Parameter space declaration.

// Start of header guard
#ifndef NETWORK_H
#define NETWORK_H

// This function creates a network given a set of parameters
// for network generation using a Barabasi algorithm. The function
// returns the number of connections in the network.
uint32_t createNetwork(igraph_matrix_t* h_network,
                       const Parameter h_parameter,
                       const uint32_t h_subPopTotal);

// This function extracts the information from a sparse adjacency matrix
// in igraph and converts it into a compressed sparse row matrix of with types
// float [NNZ] (weights),  int [|V| + 1] (locations in float[NNZ] that start a row), 
// and int [NNZ] (column indices of non-zero weights).
void igraphSparseToCSR(float* h_sparseNetWeight,
                       uint32_t* h_sparseNetTo,
	                   uint32_t* h_sparseNetLoc,
	                   const igraph_matrix_t h_network,
                       const uint32_t h_subPopTotal);

// This function gets the number of connections of each node
// and generates the cumulative expected relative size of each community
// in the network.
void calcCumulativeComSize(float* h_nCumulativeComSize,
                           float* h_mCumulativeComSize,
                           const uint32_t* h_sparseNetLoc,
                           const SimulationParam h_simulationParam,
                           const uint32_t h_subPopTotal);

// This function returns the subpopulation number of the most well connected (urban) community
// and the last least connected (rural) community in the adjacency matrix.
void urbanRuralSubPop(uint32_t* h_urbanRuralSubPop,
                      const uint32_t* h_sparseNetLoc,
                      const uint32_t h_subPopTotal);

#endif