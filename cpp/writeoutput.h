
#include <cstdint>
#include "censustypedef.h"
#include "Parameter.h"

// Start of the header guard.
#ifndef WRITEOUTPUT_H
#define WRITEOUTPUT_H

// Writes the time-series of incidence to file.
void writeIncidence(const uint32_t* d_nIncidence,
                    const uint32_t* d_nInfectedCount,
                    const uint32_t* d_nOneSubPopInfectedCount,
                    const uint32_t* d_nCount,
                    const IOParam h_ioParam,
                    const uint32_t h_simRun,
                    const uint32_t h_mcmcRun,
                    const uint32_t maxTime);

// Writes a partial human census to file.
void writeCensus(age* d_nAge,
                 history* d_nHistory,
                 IOParam h_ioParam,
                 uint32_t h_nSize,
                 uint32_t h_simRun,
                 uint32_t h_mcmcRun);

// Writes the ages of the most recent novel infections to file.
void writeAOI(uint32_t* d_nAgeOfInfection,
              uint32_t h_simRun,
              uint32_t h_mcmcRun,
              IOParam h_ioParam);

// Writes the timeseries of incidence in each subpopulation to file.
void writeSpatial(uint32_t* d_nInfectedSubPopSeries,
                  uint32_t h_subPopTotal,
                  uint32_t maxTime,
                  uint32_t h_simRun,
                  uint32_t h_mcmcRun,
                  IOParam h_ioParam);

#endif