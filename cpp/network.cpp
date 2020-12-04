// network.cu: constructs the complex network of a given number
// of nodes and a suggested number of connections between each node.

#include <fstream>				// Output file stream.
#include "igraph.h"				// Generating complex networks.
#include "Parameter.h"		    // Parameter space definition.
#include "constant.h"			// Constants for simulation.

// This function creates a network given a set of parameters
// for network generation using a Barabasi algorithm. The function
// returns the number of connections in the network.
uint32_t createNetwork(igraph_matrix_t *h_network,
					   const Parameter h_parameter,
                       const uint32_t h_subPopTotal)
{
	// Declare a graph/network in igraph.
	igraph_t graph;

	// Check if the human and mosquito population is to be organized
	// into a complex network or a non-wrapping lattice.
	if (C_NETWORK > 0)
	{
		// Choose a seed for graph generation and create the network using a
		// barabasi algorithm.
		igraph_rng_seed(igraph_rng_default(), static_cast<unsigned long>(h_parameter.simulation.netSeed));
		igraph_barabasi_game(&graph, h_subPopTotal, h_parameter.simulation.netPower,
			static_cast<int>(h_parameter.simulation.netM), NULL, false, 1.0f, 0, IGRAPH_BARABASI_PSUMTREE, NULL);
	}
	else
	{
		// Create a vector containing the dimensions of the lattice.
		igraph_vector_t v;
		igraph_vector_init(&v, 2);
		igraph_vector_set(&v, 0, pow(h_parameter.simulation.metaPop, 0.5f));
		igraph_vector_set(&v, 1, pow(h_parameter.simulation.metaPop, 0.5f));

		// Create the lattice graph.
		igraph_lattice(&graph, &v, 1, false, false, false);

		// Destroy the dimension vector.
		igraph_vector_destroy(&v);
	}

	// Initialize the sparse matrix and get the adjacency (sparse) matrix for the network.
	igraph_matrix_init(h_network, h_subPopTotal, h_subPopTotal);
	igraph_get_adjacency(&graph, h_network, IGRAPH_GET_ADJACENCY_BOTH, 0);

	// Get the number of edges in the network.
	uint32_t h_netNumEdges;
	h_netNumEdges = igraph_ecount(&graph);

	// Write the adjacency matrix to file.
	/*std::ofstream networkData(h_parameter.io.get_odir() + "/network.csv");
	for (int i = 0; i < h_subPopTotal; ++i)
	{
		for (int j = 0; j < h_subPopTotal; ++j)
		{
			networkData << ((j == 0)?"":",") << MATRIX(*h_network, i, j);
		}
		networkData << "\n";
	}
	networkData.close();*/

    // Free the graph.
    igraph_destroy(&graph);    

	// Return the total number of connections (NNZs) in the sparse matrix.
	return h_netNumEdges;
}


// This function extracts the information from a adjacency matrix
// in igraph and converts it into a compressed sparse row matrix of with types
// float [NNZ] (weights),  int [|V| + 1] (locations in float[NNZ] that start a row), 
// and int [NNZ] (column indices of non-zero weights).
void igraphSparseToCSR(float* h_sparseNetWeight,
                       uint32_t* h_sparseNetTo,
					   uint32_t* h_sparseNetLoc,
					   const igraph_matrix_t h_network,
                       const uint32_t h_subPopTotal)
{
	// Initialize a counter for the number of non-zero weights in the 
	// adjacency matrix.
	uint32_t nnzEdgeCount = 0;

	// Store the location of the first row in the weight vector.
	h_sparseNetLoc[0] = nnzEdgeCount;

    float entry;

	// For every row and column of the adjacency matrix store
	// non zero weights in an array, the locations at which the weights 
	// for each new row begin in the array, and the column indices
	// associated with each weight.
	for (uint32_t i = 0; i < h_subPopTotal; ++i)
	{
		for (uint32_t j = 0; j < h_subPopTotal; ++j)
		{
            // Get the matrix entry.
            entry = static_cast<float>(MATRIX(h_network, i, j));

			// For every non-zero weight, store the weight and the 
			// column index, and increment the edge counter.
			if (entry > 0.95)
			{
				h_sparseNetWeight[nnzEdgeCount] = entry;
				h_sparseNetTo[nnzEdgeCount++] = j;
			}
		}
		// Store the location at which the next row begins
		// in the weight array.
		h_sparseNetLoc[i + 1] = nnzEdgeCount;
	}
}

// This function gets the number of connections of each node
// and generates the cumulative expected relative size of each community
// in the network.
void calcCumulativeComSize(float* h_nCumulativeComSize,
                           float* h_mCumulativeComSize,
                           const uint32_t* h_sparseNetLoc,
                           const SimulationParam h_simulationParam,
                           const uint32_t h_subPopTotal)
{
    // Declare and initialize the degrees of each vertex.
    uint32_t* degrees = new uint32_t[h_subPopTotal];

    // Get the degrees of each vertex in the graph.
    for (uint32_t subPop = 0; subPop < h_subPopTotal; ++subPop)
    {
        degrees[subPop] = h_sparseNetLoc[subPop + 1] - h_sparseNetLoc[subPop];
    }

    // Declare the relative community sizes.
    float* h_nRelComSize = new float[h_subPopTotal];
    float* h_mRelComSize = new float[h_subPopTotal];

    // Pass all degrees through the population size heterogeneity function
    for (uint32_t subPop = 0; subPop < h_subPopTotal; ++subPop)
    {
        h_nRelComSize[subPop] = static_cast<float>(pow(degrees[subPop], h_simulationParam.nHeteroPopSize));
        h_mRelComSize[subPop] = static_cast<float>(pow(degrees[subPop], h_simulationParam.mHeteroPopSize));
    }

    // Add up the relative size of each community plus all those that came before it.
    h_nCumulativeComSize[0] = h_nRelComSize[0];
    h_mCumulativeComSize[0] = h_mRelComSize[0];
    for (uint32_t subPop = 1; subPop < h_subPopTotal; ++subPop)
    {
        h_nCumulativeComSize[subPop] = h_nCumulativeComSize[subPop - 1] + h_nRelComSize[subPop];
        h_mCumulativeComSize[subPop] = h_mCumulativeComSize[subPop - 1] + h_mRelComSize[subPop];
    }
    
    // Normalize the cumulative sizes, such that the last entry is 1.0f.
    for (uint32_t subPop = 0; subPop < h_subPopTotal; ++subPop)
    {
        h_nCumulativeComSize[subPop] = h_nCumulativeComSize[subPop] / h_nCumulativeComSize[h_subPopTotal - 1];
        h_mCumulativeComSize[subPop] = h_mCumulativeComSize[subPop] / h_mCumulativeComSize[h_subPopTotal - 1];
    }

    delete[] h_nRelComSize;
    delete[] h_mRelComSize;
    delete[] degrees;
}

// This function returns the subpopulation number of the most well connected (urban) community
// and the last least connected (rural) community in the adjacency matrix.
void urbanRuralSubPop(uint32_t* h_urbanRuralSubPop,
                      const uint32_t* h_sparseNetLoc,
                      const uint32_t h_subPopTotal)
{
    // Declare degree of a vertex and initialize the maximum and minimum degree of the network.
    uint32_t degree, maxDegree = 0, minDegree = h_subPopTotal;

    // Declare the urban and rural subpopulation indices.
    uint32_t urbanSubPop, ruralSubPop;

    // Get the degrees of each vertex in the graph.
    for (uint32_t subPop = 0; subPop < h_subPopTotal; ++subPop)
    {
        degree = h_sparseNetLoc[subPop + 1] - h_sparseNetLoc[subPop];
        
        // Check if the subpopulation is the new rural or urban subpopulation.
        if (degree >= maxDegree)
        {
            maxDegree = degree;
            urbanSubPop = subPop;
        }
        if (degree <= minDegree)
        {
            minDegree = degree;
            ruralSubPop = subPop;
        }
    }

    // Write the urban and rural subpopulation indices to the result memory.
    h_urbanRuralSubPop[0] = urbanSubPop;
    h_urbanRuralSubPop[1] = ruralSubPop;
}