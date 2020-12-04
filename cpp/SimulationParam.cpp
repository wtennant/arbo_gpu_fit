// SimulationParam.cpp: functions for simulation parameters, including
// the default constructor and parameter sweep function.

#include <iostream>             // Input-output stream.
#include <random>               // Random number generators.
#include "SimulationParam.h"    // GPU architectural properties definition.
#include "ClimateData.h"        // Climate data class.
#include "constant.h"           // CUDA functions for getting GPU properties.

// Gets the architectural properties of the GPU. 
SimulationParam::SimulationParam()
{
    // Non-epidemiological parameters.
    nSize = 100000.0f;                                  // Number of human individuals in the metapopulation.
    metaPop = 10000.0f;                                 // Number of communities in the metapopulation.
    minMosToHuman = 1.5f;                               // Minimum mosquito to human ratio.
    mosNetGrowth = 0.5f;                                // Net growth rate of mosquitoes.
    nShapeInfantMortality = 0.4f;                       // Human life-expectancy bi-weibull scale parameter (burn in).
    nScaleInfantMortality = 1.0f / 100000.0f / C_YEAR;  // Human life-expectancy bi-weibull shape parameter (burn in).
    nScaleLifeExpectancy = 75.0f*C_YEAR;                // Second (decay) human bi-weibull scale parameter. "Close to" life expectency.
    nShapeLifeExpectancy = 6.0f;                        // Second (decay) human-bi-weibull shape parameter.
    nLocWeibull = 8.0f*C_YEAR;                          // Age at which human life-expectancy that burn in distribution becomes decay out.
    mScaleLifeExpectancy = 23.0f;                       // Mosquito life-expectancy Weibull scale parameter.
    mShapeLifeExpectancy = 4.0f;                        // Mosquito life-expectancy Weibull shape parameter.

    // Network parameters.
    netPower = 1.0f;               // Power of preferential treatment.
    netM = 1.0f;                   // Number of connections to make at each step in the Barabasi algorithm.
    netSeed = 1.0f;                // Seed for network generation.
    nHeteroPopSize = 0.0f;         // Heterogeneity parameter in human community size.
    mHeteroPopSize = 0.0f;         // Heterogeneity parameter in mosquito community size.

    // Epidemiological parameters.
    initialSeroPrev = 0.0f;             // Initial sero-prevalence of each strain in the human poplation.
    mInitialInf = 0.0f;                 // Initial mosquitoes that are infected.
    nInitialInf = 0.0f;                 // Initial humans that are infected.
    bitingRate = 0.25f;                 // The per day biting rate of mosquitoes.
    mnBitingSuccess = 0.5f;             // The probability of virus being transmitted from an infectious individual given a bite.
    nmBitingSuccess = 0.5f;             // The probability of virus being transmitted from an infectious individual given a bite.
    recovery = 4.0f;                    // The number of days humans are infectious.
    mExposed = 6.0f;                    // The number of days mosquitoes are infected, but not infectious (EIP).
    nExposed = 7.0f;                    // The number of days humans are infected, but not infectious.
    externalInfection = 5.0f;           // Imported infections per day per strain.
    longDistance = 0.01f;               // The probability of a single infectious causing long distance transmission.
    exIncPeriodRange = 2.0f;            // Maximum difference in mean EIP in off/on-season with the mid-season.
    noDistance = 0.75f;                 // The probability of a single local infection not dispersing to another community.
}

// Function that corrects the total number of sub-populations if a lattice is given, but is not square.
void SimulationParam::squareLattice()
{
    // If a lattice is requested, and the requested number of communities is
    // not a square number, adjust the number of sub-populations to the nearest
    // square number.
    if (C_NETWORK == 0)
    {
        // The two nearest square numbers.
        uint32_t upSquare = pow(ceil(pow(metaPop, 0.5)), 2.0);
        uint32_t downSquare = pow(floor(pow(metaPop, 0.5)), 2.0);

        // Difference of the two square numbers with requested sub-population size.
        uint32_t upDiff = abs(metaPop - upSquare);
        uint32_t downDiff = abs(metaPop - downSquare);

        // Find the smallest difference, and select that square number as the
        // total number of communities in the lattice.
        if ((upDiff >= downDiff) && (upDiff != 0))
        {
            metaPop = downSquare;

            // Output a warning, informing the user the number of meta-populations
            // has been changed.
            std::cout << "\rWarning: lattice dimension has been adjusted to include " <<
                pow(metaPop, 0.5) << " x " << pow(metaPop, 0.5) <<
                " = " << metaPop << " communities." << std::endl;
        }
        else if ((downDiff > upDiff) && (downDiff != 0))
        {
            metaPop = upSquare;

            // Output a warning, informing the user the number of meta-populations
            // has been changed if it is the first simulation.
            std::cout << "\rWarning: lattice dimension has been adjusted to include " <<
                pow(metaPop, 0.5) << " x " << pow(metaPop, 0.5) <<
                " = " << metaPop << " communities." << std::endl;
        }
    }
}

// Function that adjusts the parameters of the dengue simulation at each time step
// given climate data.
void SimulationParam::adjustClimate(const ClimateData climateData,
                                    const uint32_t t)
{
    // Compute the temperature in Kelvin.
    double temp_kelvin = climateData.temperature[t] + 273.15;

    // Adjust the extrinsic incubation period if
    // the information exists.
    mExposed = 24 * 0.003359 * (temp_kelvin / 298.0) * exp((15000.0 / 1.986) * ((1.0 / 298.0) - (1.0 / temp_kelvin)));
    mExposed = 1.0 / (mExposed*scaleEIP);

    // Adjust the vector life expectancy if
    // the information exists.
    mScaleLifeExpectancy = 0.8692f - 0.159f*climateData.temperature[t] +
        0.01116f*powf(climateData.temperature[t], 2.0) - 0.0003408f*powf(climateData.temperature[t], 3.0) +
        0.000003809*powf(climateData.temperature[t], 4.0); 
    mScaleLifeExpectancy = 1.0 / (mScaleLifeExpectancy*scaleMLE*powf(1 - climateData.humidity[t], scaleHumidity));
    
    // Adjust the vector to human transmission success.
    if (climateData.temperature[t] < 17.05)
    {
        mnBitingSuccess = 0;
    }
    else if (climateData.temperature[t] > 35.83)
    {
        mnBitingSuccess = 0;
    }
    else
    {
        mnBitingSuccess = 0.000849*climateData.temperature[t]*(climateData.temperature[t] - 17.05)*sqrt(35.83 - climateData.temperature[t]);
    }

    // Adjust the human to vector transmission success.
    if (climateData.temperature[t] < 12.22)
    {
        nmBitingSuccess = 0;
    }
    else if (climateData.temperature[t] > 37.46)
    {
        nmBitingSuccess = 0;
    }
    else
    {
        nmBitingSuccess = 0.000491*climateData.temperature[t]*(climateData.temperature[t] - 12.22)*sqrt(37.46 - climateData.temperature[t]);
    }
}