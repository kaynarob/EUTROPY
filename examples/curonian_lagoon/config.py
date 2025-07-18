# Eutropy model for n-dimentional configuration - X box(es)


import numpy as np
from utils import read_interpolate

# Simulation options
parallel_option = True
number_of_threads = 10
cache_option = True
sediment_option = True

# Depth of computational box(es)/layer(s)
H1 = 0.015    # Aerobic layer depth for all the modeling domain
H2 = 0.20     # Anaerobic layer depth for all the modeling domain

H_boxes =   {"1"    : 13.38937143,       # Average water depth for box 1
             "2"    : 2.756888889,       # Average water depth for box 2
             "3"    : 2.122842958,       # Average water depth for box 3
             "4"    : 11.23514118,       # Average water depth for box 4
             "5"    : 2.489162195,       # Average water depth for box 5
             "6"    : 3.416550000,       # Average water depth for box 6
             "7"    : 10.82673673,       # Average water depth for box 7
             "8"    : 3.139959701,       # Average water depth for box 8
             "9"    : 1.949559459,       # Average water depth for box 9
             "10"   : 11.96268889,       # Average water depth for box 10
             "11"   : 8.060419565,       # Average water depth for box 11
             "12"   : 20.63015769,       # Average water depth for box 12
             "13"   : 13.40079870,       # Average water depth for box 13
             "14"   : 2.037504918,       # Average water depth for box 14
             "15"   : 2.222292593,       # Average water depth for box 15
             "16"   : 9.411515000,       # Average water depth for box 16
             "17"   : 2.046905000,       # Average water depth for box 17
             "18"   : 1.310456818,       # Average water depth for box 18
             "19"   : 1.517068519,       # Average water depth for box 19
             "20"   : 2.552826667,       # Average water depth for box 20
             "21"   : 2.906471053,       # Average water depth for box 21
             "22"   : 1.978306061,       # Average water depth for box 22
             "23"   : 3.436595455,       # Average water depth for box 23
             "24"   : 2.288530435,       # Average water depth for box 24
             "25"   : 3.034168182,       # Average water depth for box 25
             "26"   : 1.645602956,       # Average water depth for box 26
             "27"   : 2.955855769,       # Average water depth for box 27
             "28"   : 3.885259302,       # Average water depth for box 28
             "29"   : 2.523247598        # Average water depth for box 29
             }


# Simulation configurations
Altitude = 0.0000                 # Site specific altitude (m).
sim_start_date = "2012-01-01"     # Simulation start date
sim_end_date = "2017-01-01"       # Simulation end date
JDay_start_date = "2012-01-01"    # Input starting Julian day
dt = 1 / 240                      # time step in days
box_out = [19]                    # List of box numbers for simulation results to be saved 
ifObs = True                      # Set it to True or False according to observation data availability
box_obs  = [19]                   # List of box numbers where the observation located
calib_sDate = '2015-01-01'        # Starting date for calibration data
valid_sDate = '2014-01-01'        # Starting date for validation data


# bentho-pelagic model constants file names
kmc_file_name1 = "input/constants_pelagic_1.txt"    # model parameter file name and path
kmc_file_name2 = "input/constants_pelagic_2.txt"    # model parameter file name and path
smc_file_name1 = "input/constants_sediment_1.txt"   # model parameter file name and path
smc_file_name2 = "input/constants_sediment_2.txt"   # model parameter file name and path

# initial concentrations for bentho-pelagic state variables
file_name_initC    = 'input/initial_concentrations.csv'     # Initial concentrations for pelagic compartment
file_name_initC_S1 = 'input/initial_concentrations_S1.csv'  # Initial concentrations for aerobic layer
file_name_initC_S2 = 'input/initial_concentrations_S2.csv'  # Initial concentrations for anerobic layer

# Boundary concentration time series
fName_df_input_NE    = "input/bc_concentration_Nemunas"      # Boundary concentration-1
fName_df_input_BS    = "input/bc_concentration_BS_average"   # Boundary concentration-2
fName_df_input_MI    = "input/bc_concentration_Minija"       # Boundary concentration-3
fName_df_input_DE    = "input/bc_concentration_Deima"        # Boundary concentration-4
fName_df_input_MA    = "input/bc_concentration_Madrosovka"   # Boundary concentration-5

# Model inputs(time series)
fName_df_input_Q_box = "input/flux_2012-2022"                # Internal water fluxes
fName_T_boxes        = "input/temp_2012-2022"                # Temperature time-series for each box
fName_V_boxes        = "input/volume_2012-2022"              # Volume time-series for each box
fName_Ia_boxes       = "input/srad_2012-2022"                # Solar radiation time-series for each box
fName_Salt_boxes     = "input/salt_2012-2022"                # Salinity time-series for each box
fName_fDay_boxes     = "input/Fraction_daylight_2012-2023"   # Fraction of daylight time-series for each box

# ============================ Model parameter \ ============================ #
kmc_keys = read_interpolate.kmc_keys
smc_keys = read_interpolate.smc_keys

# Definition of parameter sets to be used by variyig spatially
kmc1 = read_interpolate.kmc_reader(kmc_file_name1)  # pelagic compartment parameters 1
kmc2 = read_interpolate.kmc_reader(kmc_file_name2)  # pelagic compartment parameters 2
smc1 = read_interpolate.kmc_reader(smc_file_name1)  # sediment compartment parameters 1
smc2 = read_interpolate.kmc_reader(smc_file_name2)  # sediment compartment parameters 2

# NumPy array in the order of `kmc_keys`
kmc1 = np.array([kmc1[key] for key in kmc_keys])
kmc2 = np.array([kmc2[key] for key in kmc_keys])

# NumPy array in the order of `smc_keys`
smc1 = np.array([smc1[key] for key in smc_keys])
smc2 = np.array([smc2[key] for key in smc_keys])

# Definition of pelagic model parameters for desired boxes
pmcb =  {"1"    : kmc1,
         "2"    : kmc2,
         "3"    : kmc2,
         "4"    : kmc1,
         "5"    : kmc2,
         "6"    : kmc2,
         "7"    : kmc1,
         "8"    : kmc2,
         "9"    : kmc2,
         "10"   : kmc1,
         "11"   : kmc1,
         "12"   : kmc1,
         "13"   : kmc1,
         "14"   : kmc1,
         "15"   : kmc1,
         "16"   : kmc1,
         "17"   : kmc1,
         "18"   : kmc1,
         "19"   : kmc1,
         "20"   : kmc1,
         "21"   : kmc1,
         "22"   : kmc1,
         "23"   : kmc2,
         "24"   : kmc1,
         "25"   : kmc1,
         "26"   : kmc1,
         "27"   : kmc2,
         "28"   : kmc2,
         "29"   : kmc2}

# Definition of sediment model parameters for desired boxes
smcb =  {"1"    : smc1,
         "2"    : smc2,
         "3"    : smc2,
         "4"    : smc1,
         "5"    : smc2,
         "6"    : smc2,
         "7"    : smc1,
         "8"    : smc2,
         "9"    : smc2,
         "10"   : smc1,
         "11"   : smc1,
         "12"   : smc1,
         "13"   : smc1,
         "14"   : smc1,
         "15"   : smc1,
         "16"   : smc1,
         "17"   : smc1,
         "18"   : smc1,
         "19"   : smc1,
         "20"   : smc1,
         "21"   : smc1,
         "22"   : smc1,
         "23"   : smc2,
         "24"   : smc1,
         "25"   : smc1,
         "26"   : smc1,
         "27"   : smc2,
         "28"   : smc2,
         "29"   : smc2}

# ============================ Model parameter / ============================ #