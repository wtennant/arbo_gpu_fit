# Dengue individual based model code documentation
The spatially-explicit individual based model (outlined in `model.pdf`) was implemented initially in C/C++, where individuals are characterised in vectors of [Census Data](#census-variables). Populations of human and mosquito individuals are first initialised and then at each time step, individuals are passed through demographic and epidemiological processes. Throughout the simulation, numerous counters are kept track of to save outputting the entire census at every time step. These are outlined in [Counters](#counter-variables). Please refer to source files for additional details; all source files are thoroughly commented throughout.

#### Table of contents
1. [Model implementation](#dengue-individual-based-model-code-documentation)
    1. [Initialisation](#initialisation)
    2. [Demographic process](#demographic-process)
    3. [Epidemiological process](#epidemiological-process)
    4. [GPU-acceleration](#gpu-acceleration)
    5. [Compilation](#compilation)
2. [Parameters](#parameters)
3. [Constants](#constants)
4. [Variables](#variables)
    1. [Census variables](#census-variables)
    2. [Counter variables](#counter-variables)
    3. [Other variables](#other-variables)
	
## Initialisation
At the start of every simulation, the human and mosquito populations are created in the following way:
1. The ages of humans and mosquitoes were initialised according to the survival function of human and mosquito demography (i.e. demography is at equilibrium).
2. All mosquitoes were initialised to be alive at the start of the simulation.
3. Humans and mosquitoes were uniformly distributed across all communities in the meta-population.
4. All humans and mosquitoes were first initialised to be susceptible, then a small proportion of humans and mosquitoes are infected.

## Demographic process
For every individual:
1. Check if individual is alive. This is applicable to mosquitoes only, as their population size can vary.
2. Check the individual's current age against their life expectancy:
    1. if the individual has exceeded their life expectancy, then they are removed. If it is a human, then all census data for that individual are reset and new life expectancy generated. If it is a mosquito, then they are marked as being dead.
    2. if the individual has exceeded their life expectancy then their age is increased by one.
For mosquitoes, the total number of alive individuals is then compared to the expected number of individuals given the pre-defined seasonality function of mosquito density for the model. If the number is lower than expected, then new mosquitoes are created in order to match the expectation.
3. Check if the individual is infected:
    1. if the individual's age has exceeded the age at which they were due to become infectious (as in [Epidemiological process](#epidemiological-process)), make them infectious.
    2. if the individual's age has exceeded the age at which they were due to recover (humans only), remove their infection.

## Epidemiological process
For every sub-population / strain combination:
1. Calculate the total number of expected transmission events from humans to mosquitoes and from mosquitoes to humans according to the force of infection term multiplied by the total number of individuals alive in that sub-population.
2. Split the expected number of transmission events into long-distance and local transmission events.
3. For each long-distance transmission event, randomly assign it to any community within the meta-population (uniform).
4. For each local disease transmission event, randomly assign it to any community defined by the local disease dispersal kernel.

For every sub-population:
1. Go through each expected transmission event for each serotype (one-by-one), and select a random individual belonging to that sub-population.
2. If that individual is alive, not immune to the infecting serotype and not currently infected, infect the individual. Record the age of infection, and assign an age at which the individual becomes infectious and recovers (if human), record a successful transmission event.
3. If that individual is alive, but immune, or currently infected, record an unsuccessful transmission event.
4. If that individual is not alive, try again.

## GPU-acceleration
The model was implemented in NVIDIA's GPU acceleration environment: CUDA. Due to the novelty of using GPU-acceleration in epidemiological models, it is worth briefly explaining the general principles of GPU-acceleration.

### Motivation
Individual based models are highly computationally expensive because at each time step, every individual (possibly millions) needs to be passed through some demographic and/or epidemiology process. This results in model run-times being very long (on the order of several minutes for large numbers of individuals). This limits their usefulness in real-time responses to epidemiological outbreaks when often many simulations are computed. However, the graphics processing unit (GPU) is very good at processing a large number of arithmetic tasks simultaneously. Therefore, this modelling framework was implemented using GPU-acceleration because of the parallelisable nature of the demographic and epidemiological processes outlined above, whereby each individual or sub-population can be processed simultaneously.

### Implementation
In the CUDA environment, a process can be split across multiple threads (of execution). Often, these threads can be executed in parallel to one another (i.e. simultaneously). In CUDA, these threads are (conceptually) grouped into 3D blocks, which are (conceptually) organised into a 3D grid. The maximum dimensions of each block and the grid depends on the GPU architecture of the machine running the code. For the purposes of this model, the grid and blocks are only one dimensional (i.e. an array of threads).

The general approach to implementing GPU-acceleration on each process listed above is as follows:
1. Identify the level at which the process can be parallelised. For the demographical process, this is at the individual level. For the epidemiological process, this is at the sub-population/strain level.
2. Convert existing functions to GPU-accelerated code:
    1. append `__global__` to the function so that the compiler knows this is a function to operate on the GPU, to be initiated by the central processing unit (CPU).
    2. uniquely identify the thread (i.e. individual or sub-population/strain) at the start of the function using `block.Idx`, `blockDim.x` and `thread.Idx`.
    3. re-write function to operate on a single thread (i.e. individual or sub-population/strain).
4. Append calls to GPU-accelerated functions with `<<< gridDim, blockDim >>>` before the function name, where `gridDim` and `blockDim` specify the dimensions of the grid (in blocks) and each block (in threads) respectively.
5. Optimise functions to ensure that a maximal number of threads are operating concurrently.

### Optimisation
GPU-accelerated code is straight-forward to implement (provided there is existing serialised code). However, GPU-accelerated code generally only executes faster if several optimisations have been implemented. Optimisation ensures that the number of threads executing in parallel at a given time is maximised and are running as fast as possible. This number is almost always less than (otherwise equal to) the maximum number of threads that can execute concurrently, which is determined by the GPU architecture of the machine running the code. To maximise this number, it is crucial to identify the reasons behind why some threads are inactive. In general, thread inactivity is due to waiting for other threads to complete tasks before the inactive threads can continue (or even start). The exact reasons for waiting though are usually code-specific. For this model, the following optimisations were implemented:

1. **Minimise warp divergence:** GPU-acceleration with CUDA works based on the single-instruction-multiple-threads (SIMT) principle. This means that a single instruction is passed to multiple threads at once. In the case of CUDA, a single instruction is passed to 32 threads at a time, known as a warp. If any path of the 32 threads in the warp diverge from one another, through conditional branching statements (e.g `if`), then single instructions will be passed to each branch one at a time. Nested conditional statements can therefore result in the severe warp divergence with single instructions being passed to only a handful of threads at a time, while others are left waiting. It is therefore important to minimise warp divergence.
In the case of this model, it is challenging to avoid conditional statements. For example, it is necessary to check whether an individual has reached their life expectancy at each time step. However, it is useful to avoid branching over the same conditional statement multiple times throughout the code. This is to ensure that simulation run-time is not penalised multiple times throughout the code because of the same (or similar) branching process. This was originally applicable with the death and birth processes for humans. First an individual would be checked if it was due to die and only after all individuals were checked, later on in the code, a new individual was birthed in place of those who had died. This was naturally very inefficient. Instead, death and birth processes are implemented simultaneously, where a new individual is birthed in place of one who had died immediately. 
2. **Eliminate strided memory access:** In CUDA, memory is accessed simultaneously by each thread in a warp. The number of transactions required to access the memory depends upon the relationship between the thread index of the warp and the address of memory being accessed. If thread addresses have a clear 1-to-1 mapping to memory addresses (i.e. the k<sup>th</sup> thread accesses the k<sup>th</sup> entry in the data), then memory access is said to be coalesced. Coalesced memory access is optimal as it generally only requires a single transaction to access all information required by the warp. However, if the mapping from thread index to memory address is random or strided (i.e. the k<sup>th</sup> thread accesses the s*k<sup>th</sup> entry in the data, where s is the stride), then the number of transactions required is greater than one. Random or strided memory access should therefore be substituted with coalesced memory access. This can be done through careful and consistent memory management and thread assignment. In this model, this is applied when accessing the total number of infected individuals with strain `s` in a subpopulation `subPop`. As this data is ordered according to strain and then sub-population, i.e. `idx = s*h_subPopTotal + subPop`, then thread identity should correspond directly to this index and similar variables should also have this ordering.
3. **Reduce global memory access:** In CUDA, there are six different memory types which reside on the GPU. The most important is the global memory space. All threads can access global memory, however the memory bandwidth is low. It is therefore optimal to ensure that frequently used data reside in lower memory spaces, such as registers and shared memory, and/or that it is cached in L1 or L2 caches, which have much higher bandwidths. However, the total space in registers and shared memory is fairly small. The compiler will generally place frequently used variables in memory of higher bandwidth after the first read from global memory. Read-only variables can be forced to be stored in low-level caches using the `__ldg()` function.
Reducing the total number of reads and writes to global memory is by far the best way of reducing latency. It is therefore recommended that each thread only reads and writes at most once to a variable within each function. This is not always possible, particularly if data needs to be shared between threads. In the model, this is crucial in the demographic process for mosquitoes, where it is essential the know how many individuals died before birthing new ones to ensure the correct total population size is achieved at each time step (recall that mosquito population size is seasonal). This means that all threads need to communicate with one another through the low bandwidth global memory. One solution is to use shared memory. Shared memory is low latency memory that each thread on a block (see [Implementation](#implementation)) can access. Instead, the expected total population size of mosquitoes per block is calculated at each time step and the total number of mosquitoes that died in that time step is communicated between the threads in the block. This way, only threads in a block need to communicate with one another about how many mosquitoes died. Although this introduces stochasticity into the overall total population size of mosquitoes, this design is where the largest speed up in simulation runs was found.

4. **Maximise warp occupancy:** In CUDA, blocks of threads are distributed amongst several streaming multiprocessors on the GPU, which are then split into the warps of 32 threads. If work is not equally distributed across all warps (due to branching processes), then some streaming multiprocessors may finish early while others are still occupied by several warps. The game is to maximise the total number of warps executing at each possible point in time, known as warp occupancy. One way maximise warp occupancy is to adjust the number of threads on a block (and consequently the number of warps on a block). The optimal number of threads per block can be worked out experimentally by comparing simulation run times with different block sizes. It should be noted however that smaller block sizes decreases the total amount of low latency memory per block. In this model, setting 128 threads per block ensures that the total number of streaming multiprocessors doing work throughout the simulation is maximised.

5. **Reduce copying between host and device:** In order for data to be used by the GPU, data needs to be copied from the host memory to the GPU's global memory. Likewise, after computation, for data to be written to file, data needs to be copied back to host memory. This process is computationally expensive. Therefore, it is generally recommended that data only be copied once from host to device and once from device to host. This may not always be applicable if there is a mixture of parallelisable and serial code. In this case, the costs associated with copying data from the device to host, executing serial code on the CPU and copying it back to the device for parallelised functions may outweigh the costs of executing the same code serially on the GPU. This dilemma applies to the model, where individual mosquitoes and humans are infected sequentially. This part of the epidemiological process would likely run faster on the CPU. However, as the infection process is at least parallelised across sub-populations, the cost of running this section of code on the GPU outweighs the high costs associated with copying individual-level data between the host and device. If the meta-population consists of only one community, then an argument could be made where copying is worthwhile.
 
6. **Reduce data output:**  This optimisation strategy need not only apply to GPU-accelerated models, but to most individual based models, where the computational cost of writing data to file can exceed the cost of the simulation itself. This problem is more apparent in a GPU-accelerated model after simulation times have been reduced. For example, in the case of this model, if the user wants to record the total number of infections per subpopulation per serotype over time for a large number of subpopulations, writing this data to file can take longer than the simulation itself. If this data is required, then this problem is unavoidable, however if only the total number of infections per serotype across the entire meta-population is required as output, it is much faster to sum across the subpopulations in the simulation rather than outputting the files and summing elsewhere. This is the principle behind the source files `datacollect.cu` and `reduction.cuh` which summarises and collates only the data which is required for output.

## Compilation
The model was compiled in Microsoft Visual Studio 2019 on Windows 10 64-bit using the compiler `nvcc` that is included in the NVIDIA CUDA Toolkit 10.1. The C/C++ only code was compiled through `nvcc` using `cl` distributed in Microsoft Visual Studio 2019. The necessary flags to pass to the compiler were as follows:

`-O2 -Xcompiler="/std:c++14 /MD /O2" --gpu-architecture=sm_61 --machine 64 -cudart static -use_fast_math`

The option for `--gpu-architecture` should be changed in accordance with the machine executing the code. A Microsoft Visual Studio 2019 solution file is provided also here.

# Parameters
Parameters for the spatially-explicit individual based model for a multi-strain vector-borne pathogen. Simulation parameters can be adjusted within `parameter.cpp`. Default parameter values are given Table 1.1 in model.pdf. Descriptions of each parameter are outlined below, along with the corresponding mathematical notation.

#### `nSize`
The number of human individuals, $N_H$, in the simulation.

#### `metaPopRows`
The number of rows of communities in the lattice meta-population.

#### `metaPopCols`
The number of columns of communities in the lattice meta-population.

#### `maxMosToHuman`
The maximum mosquito to human ratio, $M$.

#### `minMosToHuman`
The minimum mosquito to human ratio, $m$. The mosquito to human ratio follows a sinusoidal function which is maximised at time points $t = 365n$ and minimised at time points $t = 365n + 365/2$ with $n \in \left\{0, 1, 2, ...\right\}$.

#### `nShapeInfantMortality`
The burn-in shape parameter, $a_H$, for the bi-Weibull distribution of human mortality risk.

#### `nScaleInfantMortality`
The burn-in scale parameter, $b_H$, for the bi-Weibull distribution of human mortality risk.

#### `nShapeLifeExpectancy`
The fade-out shape parameter, $c_H$, for the bi-Weibull distribution of human mortality risk.
Loosely, this defines the shape of the age distribution of individuals which survive infant mortality.

#### `nScaleLifeExpectancy`
The fade-out scale parameter, $d_H$, for the bi-Weibull distribution of human mortality risk.
Loosely, this defines the mean life expectancy of an individual which survives infant mortality risk.

#### `nLocWeibull`
The age at which the burn-in phase transititions to the fade-out phase of the bi-Weibull distribution for human mortality risk, $L$.

#### `mShapeLifeExpectancy`
The shape parameter, $c_V$, for the Weibull distribution of mosquito mortality risk. Defines the shape of the age disitribution of mosquitoes, where $c_V = 1$ denotes an exponential age distribution.

#### `mScaleLifeExpectancy`
The scale parameter, $d_V$, for the Weibull distribution of mosquito mortality risk. Approximately equal to the mean life expectancy of a mosquito: $\Gamma(1 + 1/c_V)d_V$

#### `bitingRate`
The average number of bites per day of a mosquito, $\beta$.

#### `mnBitingSuccess`
The probability of a bite resulting in transmission of the pathogen from mosquito to human, $p_H$.

#### `nmBitingSuccess`
The probability of a bite resulting in transmission of the pathogen from human to mosquito, $p_V$.

#### `recovery`
The number of days that a human is infectious, $1 / \gamma$.

#### `mExposed`
The extrinsic incubation period, or the number of days which a mosquito is infected but not infectious, $1 / \epsilon_V$.

#### `nExposed`
The intrinsic incubation period, or the number of days which a human is infected but not yet infectious, $1 / \epsilon_H$.

#### `externalInfection`
The external infection rate: the number of infections per 100,000 individuals per day per strain/serotype, $\iota$.

#### `longDistance`
The probability of a single transmission event being dispersed to anywhere within the lattice, $\omega$.

#### `exIncPeriodRange`
The maximum deviation of the extrinsic incubation period from the mean extrinsic incubation period defined by `mExposed`, $\delta$.

#### `kernelStandardDeviation`
The standard deviation of the local disease dispersal kernel. Higher values correspond to transmission events from a given community dispersing further distances within the lattice community structure, $\sigma$.

# Constants
For the simulation, a frequently-used value was treated as a constant if it rarely needed altering or was not related to the mathematical description of the individual based model itself. These values were defined as constants in order to prevent hard coding of values that may in the future be adjusted. Constants were adjusted in the header file `constant.h`. Here, the constants used in the simulation.

#### `C_MAXPARARUN`
The total number of parameter sets to simulate.

#### `C_MAXSIMRUN`
The total number of simulations per parameter set to run.

#### `C_OUTPUTFOLDER`
The folder where simulation output files are saved. Note that this folder need not exist *a priori* to executing the simulation in Windows.

#### `C_SHUTDOWN`
Boolean for shutting down the computer at the end of the simulation. Used for large sensitivity analyses and vacations.

#### `C_STRAINS`
The number of strains/serotypes of the pathogen to simulate. In the case of dengue, this was set to 4.

#### `C_MMAXINITIALAGE`
The maximum mosquito age in days of the initialised mosquito population.

#### `C_NMAXINITIALAGE`
The maximum human age in year of the initialised human population.

#### `C_YEAR`
The number of days per year.

#### `C_INITIALMINTIME`
The minimum burn in period of the initial simulations. This gives outbreaks in almost entirely susceptible populations, such that $R_0$ can be calculated from the initial growth rate.

#### `C_INITIALMAXTIME`
The maximum burn in period for the initial simulations from which the user-defined simulations begin.
This ensures that seroprevalence has reached some dynamic equilibrium.

#### `C_NSIZERECORD`
The maximum number of humans in the census to record at the end of the simulation.
This prevents the large consumption of disk space in large sensitivity analyses and population sizes.

#### `C_NAOIRECORD`
The total number of ages of most recent heterotypic infections to store.

#### `C_MAXINTROATTEMPT`
The maximum number of attempts to introduce an infection from an external source (outside of the lattice).
This is done in order to prevent infinite loops in situations where seroprevalence is exceptionally high (i.e. after an initial outbreak with parameter values describing high transmissibility).

#### `C_THREADSPERBLOCK`
The total number of GPU threads assigned per virtual block when calling kernels (or functions parallelised on the device). The maximum number of threads per block is limited by the GPU architecture. Generally, powers of two above and equal to 32 is recommended. Depending upon the kernel, this number can be optimised to minimise kernel run-time such that a balance is struck between shared memory reads/writes (low-level memory reads and writes within the same block) and divergence of threads within the same block (threads may take different computational pathways: i.e. one individual may be aged, the other may be killed & birthed). In order to optimise this value, experiments were done to minimise the run time of the demographic process (see [Demographic process](#demographic-process) and [Optimisation](optimisation) for more details). More information on selecting the optimal number of GPU threads per block in a CUDA kernel can be found in the NVIDIA CUDA manual.

#### `C_THREADSPERBLOCKSUM`
Total number of GPU threads assigned per virtual block when calling reduction kernels, or kernels related to summing arrays of values.

# Variables
Copies of variables that stored on both the host and device were prefixed with `h_` and `d_` respectively.
Otherwise, variables were only stored on the host/device in which they were created, i.e. variables without the above prefix were declared on the device in functions containing the flags `__global__` and on the host in other cases.

Here, the main variables of the simulation are listed along with a brief description:

## Census variables
For human and mosquito individuals, variables were prefixed with `n` and `m` respectively. Each variable corresponded to a single piece of information about an individual, such as age, or community. Each census variables was an array of values, where the index within the array mapped to the identity of an individual. Each human variable had length as a multiple of the total number of human individuals `h_nSize`, and each mosquito variable had length as a multiple of the maximum possible number of mosquitoes `h_mSize`. The census variables used during the simulation are outlined below, note whether these apply to humans (n) and/or mosquitoes (m), and the mapping of array index to individual ID.

#### `Age` [n, m]
Age in days of each individuals.
1-to-1 mapping for array index $\to$ individual ID.

#### `Dead` [m]
Whether each mosquito is alive or dead. This is used in keeping track of fluctuations in mosquito population size.
1-to-1 mapping for array index $\to$ individual ID.

#### `Exposed` [n, m]
Age at which an infected individual becomes infectious.
1-to-1 mapping for array index $\to$ individual ID.

#### `History` [n]
Immunological history of a human individual:

1. `0` corresponds to no previous infection,
2. `65535` corresponds to infection at age zero,
3. otherwise, corresponds to age of infection.

The mapping from array index $\to$ individual ID depends upon the number of strains of pathogen. For dengue, this results in a 4-to-1 mapping. In general, $\textrm{index mod}$ `h_nSize` $= \textrm{ID}$. This means that the immunological history is sorted by strain first, and then by individual ID, where the immunological history of strain `s` of individual `idx` is contained in entry `s*h_nSize + idx`.

#### `InfectStatus` [n, m]
Infection status of each individual:

1. `0` corresponds to uninfected,
2. `1` corresponds to exposed (infected but not infectious),
3. `2` corresponds to infectious.

1-to-1 mapping for array index $\to$ individual ID.

#### `PLE` [n, m]
Random probability assigned at birth of individual. Each simulation day, this probability is compared to the cumulative probability of death evaluated at their current age in order to determine if an individual would die.
1-to-1 mapping for array index $\to$ individual ID.

#### `Recovery` [n]
Age at which an infectious individual recovers/stops being infectious.
1-to-1 mapping for array index $\to$ individual ID.

#### `Strain` [n, m]
Current/most-recent infecting strain of pathogen. For dengue, takes 0--3.
1-to-1 mapping for array index $\to$ individual ID.

#### `SubPopulation` [n, m]
Community, or sub-population, which the individual is resident to.
1-to-1 mapping for array index $\to$ individual ID.

## Counter variables
In order to reduce unnecessary data storage (by writing census daily) or computation (by calculating population totals from each census daily), counter variables were declared which kept track of the total number of individuals which matched specific epidemiological and demographical criteria. These counter variables were increased or decreased on the fly, as individuals moved between demographical and epidemiological states. Outlined below are some of the information kept track of over time.

#### `DeadCount` [m]
Counted the number of dead individuals per block of device threads. Allowed usage of lower-level memory (shared memory access across blocks of threads) on the device in order to get the correct mosquito population size at each time step (see [Optimisation](#optimisation)).

#### `SubPopCount` [n,m]
Total number of individuals within each subpopulation at the current time step only.

#### `InfectedSubPopCount` [n,m]
Total number of infected individuals within each subpopulation of each strain at the current time step only. First sorted by strain and then by subpopulation. In other words, the number of infected individuals in subpopulation `subPop` of strain `s` is in entry `s*subPopTotal + subPop`, where `subPopTotal` is the total number of subpopulations.

#### `InfectedCount` [n]
Total number of infected individuals of each strain at each time step. First sorted by time, then by strain. In other words, the total number of infection individuals of strain `s` and time `t` is in position `t*C_STRAINS + s` of the array, where `C_STRAINS` is the total number of pathogen strains/serotypes.

#### `OneSubPopCount` [n]
Total number of infected individuals of each strain at each time step within a randomly chosen sub-population. This data is later used to calculate the probability of two serotypes co-circulating within the same community at any given time. Arranged as above.

#### `Count` [n, m]
Total number of individuals at each time step.

#### `ReductionInfectedCount` and `ReductionCount` [n, m]
Used quickly calculating the total number of (infected) individuals in the entire meta-population (see [Optimisation](#optimisation)).

## Other variables
### `SubPopIndex` [n, m]
IDs of individuals arranged by subpopulation.

#### `SubPopLoc` [n, m]
Indices of `SubPopIndex` which correspond to the start of the next subpopulation.

#### `SubPopSize` [n, m]
Maximum number of individuals within each community. The above three variables are used to randomly select individuals to infect within a given subpopulation.

#### `AgeOfInfection` [n, m]
Ages of the most recent heterotypic human infections. The total number of ages for each exposure (1st, 2nd, 3rd and 4th) recorded are given by `C_NAOIRECORD` (see [Constants](#constants)).

#### `metaPopCols` and `metaPopRows`
Dimensions of the lattice community structure.

#### `subPopTotal`
Total number of communities.

#### `randStates`
The random number generator states used to generate random numbers for demographical and epidemiological processes on the device. The number of random number generator states corresponds to the total number of possible active threads on the GPU architecture (see [Optimisation](#optimisation)).
