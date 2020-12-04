// IOParam.cpp: Default constructor for input-output parameters.

#include "IOParam.h"    // GPU architectural properties definition.

// Default constructor for input-output parameters.
IOParam::IOParam()
{
    idir = "../../../Input files";
    iClimateFile = "climate_smooth.csv";
    iDengueFile = "dengue.csv";
    odir = "../../../Output files";
}

// Function which parses the input command line arguments.
void IOParam::commandLineParse(int argc, char* argv[])
{
    // Read each input argument and store if necessary.
    for (int i = 1; i < argc; i++)
    {
        // Input directory flag is given.
        if (strcmp(argv[i], "-idir") == 0)
        {
            idir = argv[i + 1];
        }

        // Input climate file flag is given.
        else if (strcmp(argv[i], "-iclimate") == 0)
        {
            iClimateFile = argv[i + 1];
            i++;
        }

        // Input dengue file flag is given.
        else if (strcmp(argv[i], "-idengue") == 0)
        {
            iDengueFile = argv[i + 1];
            i++;
        }

        // Output directory flag is given.
        else if (strcmp(argv[i], "-odir") == 0)
        {
            odir = argv[i + 1];
            i++;
        }
    }
}