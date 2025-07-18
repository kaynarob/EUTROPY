# Eutropy model for n-dimentional configuration - X box(es)


import numpy as np
from utils import read_interpolate

# Simulation options
parallel_option   = 
number_of_threads = 
cache_option      = 
sediment_option   = 

# Depth of computational box(es)/layer(s)
H1 =     # Aerobic layer depth for all the modeling domain
H2 =     # Anaerobic layer depth for all the modeling domain

H_boxes =   {" "    :  ,       # Average water depth for box n
             " "    :  ,       # Average water depth for box n
             " "    :  ,       # Average water depth for box n
             " "    :  ,       # Average water depth for box n
             " "    :  ,       # Average water depth for box n
             " "    :  ,       # Average water depth for box n
             " "    :  ,       # Average water depth for box n
             " "    :  ,       # Average water depth for box n
             " "    :  ,       # Average water depth for box n
             "  "   :  ,       # Average water depth for box n
             "  "   :  ,       # Average water depth for box n
             "  "   :  ,       # Average water depth for box n
             "  "   :  ,       # Average water depth for box n
             "  "   :  ,       # Average water depth for box n
             "  "   :  ,       # Average water depth for box n
             "  "   :  ,       # Average water depth for box n
             "  "   :  ,       # Average water depth for box n
             "  "   :  ,       # Average water depth for box n
             "  "   :  ,       # Average water depth for box n
             "  "   :  ,       # Average water depth for box n
             "  "   :  ,       # Average water depth for box n
             "  "   :  ,       # Average water depth for box n
             "  "   :  ,       # Average water depth for box n
             "  "   :  ,       # Average water depth for box n
             "  "   :  ,       # Average water depth for box n
             "  "   :  ,       # Average water depth for box n
             "  "   :  ,       # Average water depth for box n
             "  "   :  ,       # Average water depth for box n
             "  "   :          # Average water depth for box n
             }


# Simulation configurations
Altitude =                  # Site specific altitude (m).
sim_start_date = ""         # Simulation start date
sim_end_date = ""           # Simulation end date
JDay_start_date = ""        # Input starting Julian day
dt =                        # time step in days
box_out = []                # List of box numbers for simulation results to be saved 
ifObs =                     # Set it to True or False according to observation data availability
box_obs  = []               # List of box numbers where the observation located
calib_sDate = ''            # Starting date for calibration data
valid_sDate = ''            # Starting date for validation data


# bentho-pelagic model constants file names
kmc_file_name1 = " "    # model parameter file name and path
kmc_file_name2 = " "    # model parameter file name and path
smc_file_name1 = " "    # model parameter file name and path
smc_file_name2 = " "    # model parameter file name and path

# initial concentrations for bentho-pelagic state variables
file_name_initC    = ' '     # Initial concentrations for pelagic compartment
file_name_initC_S1 = ' '     # Initial concentrations for aerobic layer
file_name_initC_S2 = ' '     # Initial concentrations for anerobic layer

# Boundary concentration time series
fName_df_input_NE    = ""    # Boundary concentration-1
fName_df_input_BS    = ""    # Boundary concentration-2
fName_df_input_MI    = ""    # Boundary concentration-3
fName_df_input_DE    = ""    # Boundary concentration-4
fName_df_input_MA    = ""    # Boundary concentration-5

# Model inputs(time series)
fName_df_input_Q_box = ""   # Internal water fluxes
fName_T_boxes        = ""   # Temperature time-series for each box
fName_V_boxes        = ""   # Volume time-series for each box
fName_Ia_boxes       = ""   # Solar radiation time-series for each box
fName_Salt_boxes     = ""   # Salinity time-series for each box
fName_fDay_boxes     = ""   # Fraction of daylight time-series for each box

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
pmcb =  {" "    : kmc1,
         " "    : kmc2,
         " "    : kmc2,
         " "    : kmc1,
         " "    : kmc2,
         " "    : kmc2,
         " "    : kmc1,
         " "    : kmc2,
         " "    : kmc2,
         "  "   : kmc1,
         "  "   : kmc1,
         "  "   : kmc1,
         "  "   : kmc1,
         "  "   : kmc1,
         "  "   : kmc1,
         "  "   : kmc1,
         "  "   : kmc1,
         "  "   : kmc1,
         "  "   : kmc1,
         "  "   : kmc1,
         "  "   : kmc1,
         "  "   : kmc1,
         "  "   : kmc2,
         "  "   : kmc1,
         "  "   : kmc1,
         "  "   : kmc1,
         "  "   : kmc2,
         "  "   : kmc2,
         "  "   : kmc2}

# Definition of sediment model parameters for desired boxes
smcb =  {" "    : smc1,
         " "    : smc2,
         " "    : smc2,
         " "    : smc1,
         " "    : smc2,
         " "    : smc2,
         " "    : smc1,
         " "    : smc2,
         " "    : smc2,
         "  "   : smc1,
         "  "   : smc1,
         "  "   : smc1,
         "  "   : smc1,
         "  "   : smc1,
         "  "   : smc1,
         "  "   : smc1,
         "  "   : smc1,
         "  "   : smc1,
         "  "   : smc1,
         "  "   : smc1,
         "  "   : smc1,
         "  "   : smc1,
         "  "   : smc2,
         "  "   : smc1,
         "  "   : smc1,
         "  "   : smc1,
         "  "   : smc2,
         "  "   : smc2,
         "  "   : smc2}

# ============================ Model parameter / ============================ #