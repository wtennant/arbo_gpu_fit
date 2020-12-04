// SimulationParam.h: contains declaration of simulation parameter class.

#include <cstdint>          // Fixed-width integers.
#include <random>           // Random number generation.
#include "ClimateData.h"

// Start of the header guard.
#ifndef SIMULATIONPARAM_H
#define SIMULATIONPARAM_H

// Dengue model parameters.
class SimulationParam
{
public:
    // Demographical parameters.
    float nSize;                    // Number of human individuals in the metapopulation.
    float metaPop;                  // Number of communities in the metapopulation.
    float minMosToHuman;            // Minimum mosquito to human ratio.
    float mosNetGrowth;             // Net growth rate of mosquitoes.
    float nShapeInfantMortality;    // Human life-expectancy bi-weibull scale parameter (burn in).
    float nScaleInfantMortality;    // Human life-expectancy bi-weibull shape parameter (burn in).
    float nScaleLifeExpectancy;     // Second (decay) human bi-weibull scale parameter. "Close to" life expectency.
    float nShapeLifeExpectancy;     // Second (decay) human-bi-weibull shape parameter.
    float nLocWeibull;              // Age at which human life-expectancy that burn in distribution becomes decay out.
    float mScaleLifeExpectancy;     // Mosquito life-expectancy Weibull scale parameter.
    float mShapeLifeExpectancy;     // Mosquito life-expectancy Weibull shape parameter.

    // Network generation parameters.
    float netPower;                 // Power of preferential treatment.
    float netM;                     // Number of connections to make at each step in the Barabasi algorithm.
    float netSeed;                  // Seed for network generation.
    float nHeteroPopSize;           // Heterogeneity parameter in human community size.
    float mHeteroPopSize;           // Heterogeneity parameter in mosquito community size.

    // Epidemiological parameters.
    float initialSeroPrev;          // Initial sero-prevalence of each strain in the human poplation.
    float mInitialInf;              // Initial infected mosquitoes.
    float nInitialInf;              // Initial infected humans.
    float bitingRate;               // The per day biting rate of mosquitoes.
    float mnBitingSuccess;          // The probability of virus being transmitted from an infectious individual given a bite.
    float nmBitingSuccess;          // The probability of virus being transmitted from an infectious individual given a bite.
    float recovery;                 // The number of days humans are infectious.
    float mExposed;                 // The number of days mosquitoes are infected, but not infectious (EIP).
    float nExposed;                 // The number of days humans are infected , but not infectious.
    float externalInfection;        // Infections per 100,000 per day per strain.
    float longDistance;             // The probability of a single infection causing long distance transmission .
    float exIncPeriodRange;         // Maximum difference in mean EIP in off/on-season with the mid-season.
    float noDistance;               // The probability of a single local infection not dispersing to another community.

    // Climate scaling parameters.
    float scaleEIP;                 // Scales the extrinsic incubation period from temperature data.
    float scaleMLE;                 // Scales the mosquito life expectancy from temperature data.
    float scaleCC;                  // Non-linearly scales carrying capacity.
    float scaleHumidity;            // Non-linearly scales the effect of humidity on mosquito mortality rate.

    // Default constructor for parameters.
    SimulationParam();
    void squareLattice();
    void adjustClimate(const ClimateData climateData, const uint32_t t);
};

#endif