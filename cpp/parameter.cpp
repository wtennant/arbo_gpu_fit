// Parameter.cpp: contains the methods of the parameter class, including
// writing all important simulation parmeters to file for each simulation.

#include <fstream>          // File stream types.
#include "Parameter.h"      // Parameter class.

// Write important simulation parameters to file for each simulation run.
void Parameter::writeParameters(uint32_t h_mcmcStep)
{
    // Open the file ready for writing.
    std::ofstream parameterData;
    if (h_mcmcStep == 0)
    {
        do
        {
            parameterData.open(io.get_odir() + "/parameterData.csv", std::ofstream::out);
        } while (!parameterData.is_open());

        // Write the names of each parameter to file.
        parameterData << "MCMCStep," << "nSize," << "minMosToHuman," << "subPopTotal," << "commSize,";
        parameterData << "bitingRate," << "mnBitingSuccess," << "nmBitingSuccess," << "EIPRange," << "EIP," << "IIP," << "Recovery,";
        parameterData << "noDistance," << "EIRate," << "longDistance," << "initialSP,";
        parameterData << "mDemoScale," << "mDemoShape," << "nDemoScale," << "nDemoShape,";
        parameterData << "nInfantScale," << "nInfantShape," << "nInfToDemoLoc,";
        parameterData << "netPower," << "netM," << "netSeed," << "nHeteroPower," << "mHeteroPower";
    }
    else
    { 
        do
        {
        parameterData.open(io.get_odir() + "/parameterData.csv", std::ofstream::out | std::ofstream::app);
        } while (!parameterData.is_open());
    }

    // Write the values of each parameter to file.
    parameterData << "\n" << h_mcmcStep << "," << simulation.nSize << "," << simulation.minMosToHuman << "," << simulation.metaPop << "," << simulation.nSize / simulation.metaPop;
    parameterData << "," << simulation.bitingRate << "," << simulation.mnBitingSuccess << "," << simulation.nmBitingSuccess << "," << simulation.exIncPeriodRange;
    parameterData << "," << simulation.mExposed << "," << simulation.nExposed << "," << simulation.recovery;
    parameterData << "," << simulation.noDistance << "," << simulation.externalInfection << "," << simulation.longDistance << "," << simulation.initialSeroPrev;
    parameterData << "," << simulation.mScaleLifeExpectancy << "," << simulation.mShapeLifeExpectancy;
    parameterData << "," << simulation.nScaleLifeExpectancy << "," << simulation.nShapeLifeExpectancy;
    parameterData << "," << simulation.nScaleInfantMortality << "," << simulation.nShapeInfantMortality << "," << simulation.nLocWeibull;
    parameterData << "," << simulation.netPower << "," << simulation.netM << "," << simulation.netSeed << "," << simulation.nHeteroPopSize << "," << simulation.mHeteroPopSize;

    // Close the file after writing.
    parameterData.close();
}