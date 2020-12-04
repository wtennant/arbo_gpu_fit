// ClimateData.cpp: contains function for initializing the
// the class that contains climate data.

#include <fstream>              // File streams.
#include <sstream>              // String streams.
#include <vector>               // Definition and use of vectors.
#include "ClimateData.h"        // Parameter space definition.
#include "constant.h"           // Simulation constants.

// Function that finds the column number of .csv file, given 
// the header name of that column.
void getCol(int* colNum,
            int* nCol, 
            const std::string header,
            const std::string headCol)
{
    // Initialize the column number of the header and the total
    // number of columns.
    colNum[0] = -1;
    nCol[0] = 0;

    // Declare a string for each header name.
    std::string strHead;

    // Place the header into an input string stream, so the individual
    // headers can be separated.
    std::istringstream input(header);

    // Read in each entry in the line and compare with the desired
    // column header.
    while (std::getline(input, strHead, ','))
    {
        // If the string is quoted, remove the quotes.
        if (strHead[0] == '"')
        {
            strHead = strHead.substr(1, strHead.size() - 2);
        }

        // If the current header and desired header match, record
        // the column number of the header.
        if (strcmp(headCol.c_str(), strHead.c_str()) == 0)
        {
            colNum[0] = nCol[0];
        }

        // Increment the total number of columns (next header).
        ++nCol[0];
    }
}

// Constructor which reads in a climate data file and and
// stores the required data in the climate data class.
ClimateData::ClimateData(const std::string iClimPath)
{
    // Open climate data for reading.
    std::ifstream myFile(iClimPath, std::ifstream::in);

    // Check that opening the file was successful.
    if (myFile.is_open())
    {
        // Declare a string for a line of the file, and another string
        // for the individual entries of each line.
        std::string line, str;

        // Initialize the number of days of information in the file.
        int day = 0;

        // Get the header of the climate data file.
        std::getline(myFile, line);

        // Declare variables for the total number of columns in the file,
        // and the column numbers of each epidemiological parameter.
        int nCol, temperatureCol, humidityCol, precipitationCol;

        // In the header of the climate data, get the column number of
        // each epidemiological parameter.
        getCol(&temperatureCol, &nCol, line, "T");
        getCol(&humidityCol, &nCol, line, "H");
        getCol(&precipitationCol, &nCol, line, "P");

        // For every line of data, extract the information from the desired column.
        while (std::getline(myFile, line))
        {
            // Initialize the column number of the current file.
            int col = 0;

            // Place the current line into an input file stream in order
            // to separate the line into entries.
            std::istringstream input(line);

            // Read every entry of the current line of data.
            while (std::getline(input, str, ','))
            {
                // If the string is quoted, remove the quotes to allow
                // conversion to a floating point number.
                if (str[0] == '"')
                {
                    str = str.substr(1, str.size() - 2);
                }

                // Only store the information from the desired columns.
                if (col % nCol - temperatureCol == 0)
                {
                    temperature.push_back(std::stof(str));
                }
                else if (col % nCol - humidityCol == 0)
                {
                    humidity.push_back(std::stof(str));
                }
                else if (col % nCol - precipitationCol == 0)
                {
                    precipitation.push_back(std::stof(str));
                }

                // Increment the column number.
                ++col;
            }

            // Move onto the next line of data (the next day).
            ++day;
        }

        // Store the total number of days of information in the file.
        maxTime = day;
    }
}