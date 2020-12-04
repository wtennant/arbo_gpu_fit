// Header file for the type declarations of the human and mosquito censuses.
#include <cstdint>              // Fixed width integers.

// Start of header guard
#ifndef CENSUSTYPEDEF_H
#define CENSUSTYPEDEF_H

typedef uint16_t age;           // Inidividual age in days unlikely to exceed 65535 days.
typedef uint8_t dead;           // Boolean for if individual is dead or alive (mosquito only).
typedef uint16_t exposed;       // Age at which infection becomes infectious.
typedef int32_t history;        // Strains an individual has been infected with (human only).
typedef uint8_t infectStatus;   // Disease susceptibility (= 0), infected ( = 1) and infectiousness (= 2). 
typedef float pLifeExpectancy;  // A random probability used in determining the life expectancy of individual.
typedef uint16_t recovery;      // Age at which infection ends.
typedef uint8_t strain;         // Dengue serotype the individual is infected with.
typedef uint32_t subPopulation; // The subpopulation that the individual belongs to.

#endif