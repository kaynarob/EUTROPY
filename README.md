
![logo_Eutropy](https://github.com/user-attachments/assets/c05c99e0-3ba7-4de3-8d75-0611ff5eb493)

# Eutropy

Eutrophication, Nutrient Transport, and Recycling Model for Aquatic Systems Optimized in Python combines a bentho-pelagic kinetic modeling component with a transport subroutine built on a box model framework using Just-in-Time compilation through Numba (a Python library). 
* This approach significantly reduce computational time, particularly during model calibration and extended simulation runs. It includes comprehensive biogeochemical processes and allows integration with parameter estimation tools (such as PEST and PEST++).

## Overview

The kinetic component comprises 29 state variables characterized by 130 model constants:

* **Pelagic abiotic variables**:

  * Particulate forms: C, N, P
  * Dissolved forms: C, N, P
  * Nitrate, Ammonium, Phosphate
  * Dissolved Oxygen

* **Pelagic biotic variables**:

  * Phytoplankton carbon

* **Sediment compartment variables**:
  * Particulate forms: C, N, P
  * Dissolved forms: C, N, P
  * Nitrate, Ammonium, Phosphate

Biogeochemical processes, mainly covered are:

* Decomposition
* Mineralization
* Nitrification
* Denitrification
* Phytoplankton uptake
* Reaeration
* Settling
* Burial
* Partical mixing

## Repository Structure

* **Eutropy.py**: Main model code.
* **kinetics.py**: Bentho-pelagic kinetics code.
* **config.py**: Configuration file for the model application. The available model configurations can be defined as 0, 1, 2 and 3-dimentional.
* **README.md**: General information about the software.
* **requirements.txt**: Python dependencies.
* **LICENSE.md**: Licensing information.
* **examples\curonian_lagoon\input**: Model input data for boundary conditions, fluxes, temperatures, volumes, depth, fraction of daylight, initial concentrations, and salinity.
* **examples\curonian_lagoon\observations**: Files for model calibration and validation (observations).
* **examples\curonian_lagoon\output**: Model outputs in .csv format.
* **examples\curonian_lagoon\plot**: Scripts for visualizing the model outputs/results processing.
* **examples\curonian_lagoon\utils**: Several supporting scripts to run the model such as interpolating the input data to a given time step, saving outputs, and saving outputs witha a matching date of observations.
* **examples\curonian_lagoon\config.py**: Configuration file for the model application. The available model configurations is 2D.
* **examples\curonian_lagoon\Eutropy.py**: Main model code ready to run.
* **examples\curonian_lagoon\kinetics.py**: Bentho-pelagic kinetics are defined.


## How to Run

### Prerequisites

Install dependencies:

```bash
pip install -r requirements.txt
```

### Model Execution

* **Eutropy model**: Run `eutropy.bat` inside examples/curonian_lagoon folder or :

```bash
py -3.9 Eutropy.py
```

### Results Visualization

Web-based platform for visualizing:
* [Model outputs with statistics.](http://laguna.ku.lt:3838/sample-apps/Eutro0.2/)

Scripts for visualizing:

* **plot_output.py**: Model outputs.
* **subplot_statistics.py**: Model outputs with statistics.



## Requirements

* Python 3.9
* numba 0.59.1
* numPy 1.26.4
* pandas 2.2.2
* scipy 1.13.0
* matplotlib 3.6.3

## References
* Python Software Foundation. (2020). Python Language Reference, version 3.9. Available at https://www.python.org/.
* Lam, S. K., Pitrou, A., & Seibert, S. (2015). Numba: A LLVM-based Python JIT Compiler. Proceedings of the Second Workshop on the LLVM Compiler Infrastructure in HPC. https://doi.org/10.1145/2833157.2833162.
* Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with NumPy. Nature, 585, 357–362. https://doi.org/10.1038/s41586-020-2649-2.
* McKinney, W. (2010). Data Structures for Statistical Computing in Python. Proceedings of the 9th Python in Science Conference, 51-56. https://doi.org/10.25080/Majora-92bf1922-00a.
* Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17, 261–272. https://doi.org/10.1038/s41592-019-0686-2.
* Hunter, J.D. (2007). Matplotlib: A 2D Graphics Environment. *Computing in Science & Engineering, 9*(3), 90-95.
* White, J.T., Hunt, R.J., Fienen, M.N., and Doherty, J.E., 2020, Approaches to Highly Parameterized Inversion: PEST++ Version 5, a Software Suite for Parameter Estimation, Uncertainty Analysis, Management Optimization and Sensitivity Analysis: U.S. Geological Survey Techniques and Methods 7C26, 52 p., https://doi.org/10.3133/tm7C26.
* J. Doherty, Model-Independent Parameter Estimation User Manual. Brisbane, Australia: Watermark Numerical Computing, 2018.

### Related Links:

- [How to simplify the calibration of ecological models by using PEST and PEST++?](https://doi.org/https://doi.org/10.1016/j.ecoinf.2025.103213)
- [A simple ecological model: CuLPy](https://github.com/kaynarob/CuLPy)
- [Worked example: Calibration and uncertainty quantification of aquatic ecological models - in Python.](https://doi.org/10.5281/zenodo.15040756)

## Contact

Burak Kaynaroglu,
email: [burak.kaynaroglu@ku.lt](mailto:burak.kaynaroglu@ku.lt)
