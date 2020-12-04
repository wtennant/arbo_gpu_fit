// writeoutput.cpp: contains functions for writing simulation output to file,
// including a time series of incidence, and most recent ages of infection.

#include <cstdint>              // Fixed-width integers.
#include <fstream>              // File streams.
#include <iostream>             // Input-output streams.
#include "cuda_runtime_api.h"   // CUDA copy functions.
#include "censustypedef.h"      // Census type definitions.
#include "IOParam.h"            // Input-output parameters.
#include "constant.h"           // Simulation constants.

// Writes the time-series of incidence to file.
void writeIncidence(const uint32_t* d_nIncidence,
                    const uint32_t* d_nInfectedCount,
                    const uint32_t* d_nOneSubPopInfectedCount,
                    const uint32_t* d_nCount,
                    const IOParam h_ioParam,
                    const uint32_t h_simRun,
                    const uint32_t h_mcmcRun,
                    const uint32_t maxTime)
{
    // Declare the CPU counters for the number of infected humans in one specific subpopulation, 
    // the total number of infected humans per strain, the total number of new infections per strain,
    // and the total number of human and mosquitoes at each time step of the simulation.
    uint32_t* h_nOneSubPopInfectedCount = new uint32_t[(maxTime + 1)*C_STRAINS];
    uint32_t* h_nInfectedCount = new uint32_t[(maxTime + 1)*C_STRAINS];
    uint32_t* h_nIncidence = new uint32_t[(maxTime + 1)*C_STRAINS];
    uint32_t* h_nCount = new uint32_t[maxTime + 1];

    // Copy the time series avriables from the GPU device to the CPU host.
    cudaMemcpy(h_nOneSubPopInfectedCount, d_nOneSubPopInfectedCount, sizeof(uint32_t)*(maxTime + 1)*C_STRAINS, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_nInfectedCount, d_nInfectedCount, sizeof(uint32_t)*(maxTime + 1)*C_STRAINS, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_nIncidence, d_nIncidence, sizeof(uint32_t)*(maxTime + 1)*C_STRAINS, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_nCount, d_nCount, sizeof(uint32_t)*(maxTime + 1), cudaMemcpyDeviceToHost);

    // Open human and mosquito time series data files ready for recording, which saves all data which gets
    // recorded at every time step, for example human and mosquito population sizes, or the total
    // number of infected individuals.
    std::ofstream nData;
    if (h_simRun == 0)
    {
        nData.open(h_ioParam.get_odir() + "/nData" + std::to_string(h_mcmcRun) + ".csv", std::ofstream::out);

        // Write the headers of each column in the data file.
        uint32_t serotype{ 0 };
        nData << "t,popSize";
        while (serotype < C_STRAINS)
        {
            nData << "," << "incidence_DENV" << serotype + 1;
            ++serotype;
        }
        serotype = 0;
        while (serotype < C_STRAINS)
        {
            nData << "," << "prevalence_DENV" << serotype + 1;
            ++serotype;
        }
        serotype = 0;
        while (serotype < C_STRAINS)
        {
            nData << "," << "prevalence_DENV" << serotype + 1 << "_random_community";
            ++serotype;
        }
        nData << ",simRun";
    }
    else
    {
        nData.open(h_ioParam.get_odir() + "/nData" + std::to_string(h_mcmcRun) + ".csv", std::ofstream::out | std::ofstream::app);
    }

    // For every time step, record the time series data. 
    for (uint32_t t = 0; t <= maxTime; ++t)
    {
        nData << "\n" << t << "," << h_nCount[t];
        uint32_t serotype{ 0 };
        while (serotype < C_STRAINS)
        {
            nData << "," << h_nIncidence[t*C_STRAINS + serotype];
            ++serotype;
        }
        serotype = 0;
        while (serotype < C_STRAINS)
        {
            nData << "," << h_nInfectedCount[t*C_STRAINS + serotype];
            ++serotype;
        }
        serotype = 0;
        while (serotype < C_STRAINS)
        {
            nData << "," << h_nOneSubPopInfectedCount[t*C_STRAINS + serotype];
            ++serotype;
        }
        nData << "," << h_simRun;
    }

    // Close the data file for recording population size.
    nData.close();
}

// Writes a partial human census to file.
void writeCensus(age* d_nAge,
                 history* d_nHistory,
                 IOParam h_ioParam,
                 uint32_t h_nSize,
                 uint32_t h_simRun,
                 uint32_t h_mcmcRun)
{
    // Record the data for alive humans (used in computing R0 from immunity landscape).
    // Declare memory for humans age and immunological history.
    uint32_t h_nSizeRecord = static_cast<uint32_t>(fminf(C_NSIZERECORD, h_nSize));
    age* h_nAge= new age[h_nSizeRecord];
    history* h_nHistory = new history[h_nSize*C_STRAINS];

    // Copy from the graphics cards memory to the host.
    cudaMemcpy(h_nAge, d_nAge, sizeof(age)*h_nSizeRecord, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_nHistory, d_nHistory, sizeof(history)*h_nSize*C_STRAINS, cudaMemcpyDeviceToHost);

    // Open the file and record headers.
    std::ofstream nAlive;
    if (h_simRun == 0)
    {
        nAlive.open(h_ioParam.get_odir() + "/nAlive_" + std::to_string(h_mcmcRun) + ".csv", std::ofstream::out);
        nAlive << "age";
        for (uint32_t s = 0; s < C_STRAINS; ++s)
        {
            nAlive << ",history" << s + 1;
        }
        nAlive << ",simRun";
    }
    else
    {
        nAlive.open(h_ioParam.get_odir() + "/nAlive_" + std::to_string(h_mcmcRun) + ".csv", std::ofstream::out | std::ofstream::app);
    }

    // Record the data.
    for (uint32_t i = 0; i < h_nSizeRecord; ++i)
    {
        nAlive << "\n" << h_nAge[i];
        for (uint32_t s = 0; s < C_STRAINS; ++s)
        {
            nAlive << "," << h_nHistory[s*h_nSize + i];
        }
        nAlive << "," << h_simRun;
    }
    nAlive.close();

    // Delete the dynamically allocated variables.
    delete[] h_nAge;
    delete[] h_nHistory;
}

// Writes the ages of the most recent novel infections to file.
void writeAOI(uint32_t* d_nAgeOfInfection,
              uint32_t h_simRun,
              uint32_t h_mcmcRun,
              IOParam h_ioParam)
{
    // Declare memory for the most recent ages of infection for each novel exposure.
    uint32_t* h_nAgeOfInfection = new uint32_t[C_NAOIRECORD*C_STRAINS];

    // Copy the ages of most recent infections from the GPU device to the CPU host.
    cudaMemcpy(h_nAgeOfInfection, d_nAgeOfInfection, sizeof(uint32_t) * C_NAOIRECORD * C_STRAINS, cudaMemcpyDeviceToHost);

    // Open a file for recording the most recent ages of infections of each novel exposure.
    std::ofstream nAOI;
    if (h_simRun == 0)
    {
        // Record the headers of the file.
        nAOI.open(h_ioParam.get_odir() + "/nAOI_" + std::to_string(h_mcmcRun) + ".csv", std::ofstream::out);
        nAOI << "first,second,third,fourth";
        nAOI << ",simRun";
    }
    else
    {
        nAOI.open(h_ioParam.get_odir() + "/nAOI_" + std::to_string(h_mcmcRun) + ".csv", std::ofstream::out | std::ofstream::app);
    }

    // For each record, save the age of infection for each novel exposure.
    for (uint32_t record = 0; record < C_NAOIRECORD; ++record)
    {
        nAOI << "\n" << h_nAgeOfInfection[record];
        for (uint32_t s = 1; s < C_STRAINS; ++s)
        {
            nAOI << "," << h_nAgeOfInfection[s*C_NAOIRECORD + record];
        }
        nAOI << "," << h_simRun;
    }

    // Close the file for recording ages of infection.
    nAOI.close();
}

// Writes the timeseries of incidence in each subpopulation to file.
void writeSpatial(uint32_t* d_nInfectedSubPopSeries,
                  uint32_t h_subPopTotal,
                  uint32_t maxTime,
                  uint32_t h_simRun, 
                  uint32_t h_mcmcRun,
                  IOParam h_ioParam)
{
    // Declare memory for the time series of incidence in each subpopulation on the host.
    uint32_t* h_nInfectedSubPopSeries = new uint32_t[(maxTime + 1)*C_STRAINS*h_subPopTotal];

    // Copy the data from the GPU to the host.
    cudaMemcpy(h_nInfectedSubPopSeries, d_nInfectedSubPopSeries, sizeof(uint32_t)*h_subPopTotal*C_STRAINS*(maxTime + 1), cudaMemcpyDeviceToHost);

    // Open a file for recording spatial incidence.
    std::ofstream nSpatial;
    if (h_simRun == 0)
    {
        // Record the headers of the file.
        nSpatial.open(h_ioParam.get_odir() + "/nSpatial_" + std::to_string(h_mcmcRun) + ".csv", std::ofstream::out);        
        nSpatial << "t,subPop";
        for (uint32_t s = 0; s < C_STRAINS; ++s)
        {
            nSpatial << ",prevalence.DENV" << s + 1;
        }
        nSpatial << ",simRun";
    }
    else
    {
        nSpatial.open(h_ioParam.get_odir() + "/nSpatial_" + std::to_string(h_mcmcRun) + ".csv", std::ofstream::out | std::ofstream::app);
    }

    // For each time point, record the incidence of each subpopulation. 
    for (uint32_t t = 0; t <= maxTime; ++t)
    {
        for (uint32_t subPop = 0; subPop < h_subPopTotal; ++subPop)
        {
            nSpatial << "\n" << t << "," << subPop;
            for (uint32_t s = 0; s < C_STRAINS; ++s)
            {
                nSpatial << "," << h_nInfectedSubPopSeries[t*h_subPopTotal*C_STRAINS + s*h_subPopTotal + subPop];
            }
        }
        nSpatial << "," << h_simRun;
    }

    // Close the file.
    nSpatial.close();
}