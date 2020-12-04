// Data.h: header file containing the class definition of data,
// including the dengue and climate data.

#include "DengueData.h"     // Dengue data class.
#include "ClimateData.h"    // Climate data class.
#include "IOParam.h"        // Imput-output parameter class.

#ifndef DATA_H
#define DATA_H

class Data
{
public:
    DengueData dengue;                                              // Weekly dengue data.
    ClimateData climate;                                            // Daily climate data.
    Data(const IOParam ioParam) : dengue(ioParam.get_iDenguePath()),  // Default constructor for the data class.
        climate(ioParam.get_iClimatePath()) {};
};

#endif