// ClimateData.h: Contains the class definition for climate data.

#include <cstdint>              // Fixed-width integers.
#include <vector>               // Introduces vector types.

// Start of the header guard.
#ifndef CLIMATEDATA_H
#define CLIMATEDATA_H

// Class definition of climate data.
class ClimateData
{
public:
    uint32_t maxTime;                                   // Maximum number of days in the file.
    std::vector<float> temperature;                     // Smoothed temperature for each day in the file.
    std::vector<float> humidity;                        // Smoothed humidity for each day in the file.
    std::vector<float> precipitation;                   // Smoothed precipitation for each day in the file.
    ClimateData(const std::string);
};

#endif