// Header file containing the class definition for input and output parameters,
// including file paths.
#include <string>   // Introduces string types.

// Start of the header guard.
#ifndef IOPARAM_H
#define IOPARAM_H

// Input and output parameters.
class IOParam
{
private:
    std::string idir;												                // Input file directory.
    std::string iClimateFile;  										                // Input climate data file.
    std::string iDengueFile;                                                        // Input dengue incidence file.
    std::string odir;												                // Output file directory.

public:
    IOParam();                                                                      // Default constructor.
    std::string get_odir(void) const { return odir; }				                // Get the output directory.
    std::string get_idir(void) const { return idir; }				                // Get the input file directory.
    std::string get_iClimatePath(void) const { return idir + "/" + iClimateFile; }  // Concatenate input directory and input file to get input path.
    std::string get_iDenguePath(void) const { return idir + "/" + iDengueFile; }    // Concatenate input directory and input file to get input path.
    void commandLineParse(int argc, char* argv[]);                                  // Parse the command line to setup members.
};

#endif