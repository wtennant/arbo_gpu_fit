# Dengue simulation

# Table of contents
1. [Compilation options]
    1. [GPU Architecture]
    2. [Compile mode in MSVC]
2. [Variables]

# Compilation options

## GPU Architecture
Set the NVIDIA GPU virtual and real architecture for which the input files 
must be compiled. "compute_xx" handles the virtual, "sm_xx" handles the real 
architecture.

- Maxwell cards (e.g 960/970/980), xx=52
- Pascal cards (e.g 1060/1070/1080), xx=61

In MVS, Project Properties -> CUDA C/C++ -> Code Generation -> compute_xx,sm_xx
or use -gencode=arch=compute_xx,code=sm_xx on the command line during compilation.

## Compile mode in MSVC
It is highly recommended to compile in **x64 Release** mode in MSVC.

# Variables

## IOParam
Input/output parameter class containing input and output file directories and input file names.

```cpp
class IOParam
{
private:
	std::string idir;                                                   // Input file directory.
	std::string ifile;                                                  // Input file.
	std::string odir;                                                   // Output file directory.
public:
	IOParam() : idir(C_IDIR), ifile(C_IFILE), odir(C_ODIR) {}           // Default constructor.
	void set_idir(char* s) { idir = s; }                                // Set input file directory.
	void set_ifile(char* s) { ifile = s; }                              // Set input file name.
	void set_odir(char* s) { odir = s; }                                // Set output directory.
	std::string get_odir(void) const { return odir; }                   // Get the output directory.
	std::string get_idir(void) const { return idir; }                   // Get the input file directory.
	std::string get_ipath(void) const { return idir + "/" + ifile; }    // Concatenate input directory and input file to get input path.
};
```

## ClimParam
Climate parameter class containing climate (temperature, rainfall and precipitation) time series.

```cpp
class ClimParam
{
public:
    uint32_t maxTime;           // Maximum number of days in the file.
    std::vector<float> eip;     // Extrinsic incubation period for each day in the file.
    std::vector<float> vls;     // Vector life expectancy for each day in the file.
    std::vector<float> bite;    // Vector biting rate for each day in the file.
};
```

## Architecture
Parameter class containing properties of the local GPU architecture.

```cpp
class Architecture
{
public:
    uint32_t threadsPerWarp;    // Number of threads per warp.
    uint32_t warpsPerSM;        // Number of warps per Streaming Multiprocessor (SM).
    uint32_t totalSM;           // Total number of SMs.
};
```

## Parameter

```cpp
class Parameter
{
public:
    // Non-epidemiological parameters.
    float nSize;                    // Number of human individuals in the metapopulation.
    float metaPop;                  // Number of communities in the metapopulation.
    float maxMosToHuman;            // Maximum mosquito to human ratio.
    float minMosToHuman;            // Minimum mosquito to human ratio.
    float nShapeInfantMortality;    // Human life-expectancy bi-weibull scale parameter (burn in).
    float nScaleInfantMortality;    // Human life-expectancy bi-weibull shape parameter (burn in).
    float nScaleLifeExpectancy;     // Second (decay) human bi-weibull scale parameter. "Close to" life expectency.
    float nShapeLifeExpectancy;     // Second (decay) human-bi-weibull shape parameter.
    float nLocWeibull;              // Age at which human life-expectancy that burn in distribution becomes decay out.
    float mScaleLifeExpectancy;     // Mosquito life-expectancy Weibull scale parameter.
    float mShapeLifeExpectancy;     // Mosquito life-expectancy Weibull shape parameter.

    // Network parameters.
    float netPower;                 // Power of preferential treatment.
    float netM;                     // Number of connections to make at each step in the Barabasi algorithm.
    float netSeed;                  // Seed for network generation.
    float nHeteroPopSize;           // Heterogeneity parameter in human community size.
    float mHeteroPopSize;           // Heterogeneity parameter in mosquito community size.

    // Epidemiological parameters.
    float initialSeroPrev;          // Initial sero-prevalence of each strain in the human poplation.
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
};
```