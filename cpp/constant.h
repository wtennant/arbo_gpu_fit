// The header file containing the name space for all constants of the model.

// Start of header guard.
#ifndef CONSTANT_H
#define CONSTANT_H

// Constants external to simulation.
#define C_MAXSIMRUN 30                          // Total number of simulation runs per MCMC step.
#define C_MAXMCMCSTEPS 100000                   // Total number of steps in the MCMC chain.
#define C_SHUTDOWN 0                            // Shut down the computer at the end of the simulation.    
#define C_WRITE_SPATIAL 0                       // Write spatial data to file.
#define C_WRITE_CENSUS 0                        // Write census data to file.
#define C_WRITE_AOI 0                           // Write age of infection data to file.

// Constants explicitly internal to simulation.
#define C_NETWORK 0 							// Organise communities into complex network or not (lattice).
#define C_STRAINS 1                             // Number of serotypes.
#define C_MMAXINITIALAGE 300                    // Maximum initial mosquito age in days.
#define C_NMAXINITIALAGE 150                    // Maximum initial human age in years.
#define C_YEAR 365                              // Number of days in a year.
#define C_NSIZERECORD 5000                      // Maximum number of humans to output in a census.
#define C_NAOIRECORD 5000                       // The maximum number of ages of infection to record for each novel exposure.
#define C_MAXINTROATTEMPT 1000                  // Maximum number of external introduction attempts on susceptible individuals.
#define C_DENGUEPOPSIZE 536569.0                // Size of the population in the dengue incidence data.

// GPU-specific constants.
#define C_THREADSPERBLOCK 128                   // Number of GPU threads per GPU block per kernel call (except summing across a vector).
#define C_THREADSPERBLOCKSUM 128                // Number of threads per block per summing kernel call.

#endif