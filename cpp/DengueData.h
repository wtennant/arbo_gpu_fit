#include <cstdint>              // Fixed-width integers.
#include <vector>               // Introduces vector types.

// Start of the header guard.
#ifndef DENGUEDATA_H
#define DENGUEDATA_H

class DengueData
{
public:
    uint32_t maxWeek;                    // Maximum number of weeks in the file.
    std::vector<double> incidence;       // Human dengue incidence per week.
    DengueData(const std::string);
};

#endif