# This file is part of Eutropy
# Copyright (c) 2025 Burak Kaynaroglu
# This program is free software distributed under the MIT License 
# A copy of the MIT License can be found at 
# https://github.com/kaynarob/Eutropy/blob/main/LICENSE.md

# Eutropy model for n-dimentional configuration - X box(es)


import time
import numpy as np
import pandas as pd
from datetime import datetime
from numba import jit, set_num_threads, prange
from utils.read_interpolate import interpolate_wJDay, interpolate_wDate
from utils.save_out import save_C_to_csv, convert_C_to_dict
from utils import opt_out
import config
from kinetics import pelagic_process_rates, pelagic_with_sediment_process_rates



start_time = time.time()


parallel_option = config.parallel_option
set_num_threads(config.number_of_threads)
cache_option = config.cache_option
sediment_option = config.sediment_option
box_out = config.box_out

# =========================================================================== #
# Case specific properties, file names, initialization values /
# =========================================================================== #
H_boxes = config.H_boxes

sim_start_date = config.sim_start_date       # Simulation start date
sim_end_date = config.sim_end_date           # Simulation end date
JDay_start_date = config.JDay_start_date     # Input starting Julian day
dt = config.dt                               # time step in days
ifObs = config.ifObs                         # if there are observation data
box_obs = config.box_obs                     # in which box observation exists
calib_sDate = config.calib_sDate             # Starting date for calibration data
valid_sDate = config.valid_sDate             # Starting date for validation data

# initial concentration for each box 
df_C = pd.read_csv(config.file_name_initC)
df_C1 = pd.read_csv(config.file_name_initC_S1)
df_C2 = pd.read_csv(config.file_name_initC_S2)
statevars= list(df_C.columns[1:])
statevars_Sed1= list(df_C1.columns[1:])
statevars_Sed2= list(df_C2.columns[1:])

stateVars_toPlot = statevars

initial_Cbox_dict = {int(row['boxes']): row.drop('boxes').to_dict() for _, row in df_C.iterrows()}
initial_CS1box_dict = {int(row['boxes']): row.drop('boxes').to_dict() for _, row in df_C1.iterrows()}
initial_CS2box_dict = {int(row['boxes']): row.drop('boxes').to_dict() for _, row in df_C2.iterrows()}

boxes = [box for box in df_C["boxes"]]
# =========================================================================== #
# Case specific model properties \
# =========================================================================== #

print('\n# =================================================================== #')
print('\nmodel initialization...\n')
total_boxNo = len(H_boxes)
print(f'\tparallel_option                : {parallel_option}')
print(f'\tcache_option                   : {cache_option}')
print(f'\tnumber of boxes                : {total_boxNo}')
print(f'\tsediment_option                : {sediment_option}')
print(f'\tsimulation start date          : {sim_start_date}')
print(f'\tsimulation end date            : {sim_end_date}')
print(f'\ttime step in days              : {int(1/dt)}')
print(f'\tdt                             : {int(24*60*dt)} minute(s)')

# =========================== number of iteration =========================== #
date_1 = datetime.strptime(sim_start_date, '%Y-%m-%d')
date_2 = datetime.strptime(sim_end_date, '%Y-%m-%d')
sim_end_jdays = (date_2 - date_1).days
n_iter = int(sim_end_jdays / dt)  
# =========================== number of iteration =========================== #

# ============================ Model parameter \ ============================ #
pmcb = config.pmcb
smcb = config.smcb
# ============================ Model parameter / ============================ #

# =========================================================================== #
# ========================= Read/Interpolate Data \ ========================= #
df_input_Q_box = interpolate_wJDay(sim_start_date, sim_end_date, JDay_start_date, 1/dt, config.fName_df_input_Q_box)
T_boxes        = interpolate_wJDay(sim_start_date, sim_end_date, JDay_start_date, 1/dt, config.fName_T_boxes)
V_boxes        = interpolate_wJDay(sim_start_date, sim_end_date, JDay_start_date, 1/dt, config.fName_V_boxes)
Ia_boxes       = interpolate_wJDay(sim_start_date, sim_end_date, JDay_start_date, 1/dt, config.fName_Ia_boxes)
Salt_boxes     = interpolate_wJDay(sim_start_date, sim_end_date, JDay_start_date, 1/dt, config.fName_Salt_boxes)
fDay_boxes     = interpolate_wDate(sim_start_date, sim_end_date, 1/dt, config.fName_fDay_boxes)

df_input_NE    = interpolate_wDate(sim_start_date, sim_end_date, 1/dt, config.fName_df_input_NE)
df_input_BS    = interpolate_wDate(sim_start_date, sim_end_date, 1/dt, config.fName_df_input_BS)
df_input_MI    = interpolate_wDate(sim_start_date, sim_end_date, 1/dt, config.fName_df_input_MI)
df_input_DE    = interpolate_wDate(sim_start_date, sim_end_date, 1/dt, config.fName_df_input_DE)
df_input_MA    = interpolate_wDate(sim_start_date, sim_end_date, 1/dt, config.fName_df_input_MA)

C__1_12_BS     = {var: df_input_BS[var] for var in statevars}
C__2_24_NE     = {var: df_input_NE[var] for var in statevars}
C__3_24_MI     = {var: df_input_MI[var] for var in statevars}
C__3_3_DE      = {var: df_input_DE[var] for var in statevars}
C__4_3_MA      = {var: df_input_MA[var] for var in statevars}

bc_input_C     = {'1': C__1_12_BS, 
                  '2': C__2_24_NE, 
                  '3': C__3_24_MI, 
                  '4': C__3_3_DE, 
                  '5': C__4_3_MA}

# ========================= Read/Interpolate Data / ========================= #
# =========================================================================== #

# =========================================================================== #
# ================================= fojit \ ================================= #

n_boxes = len(initial_Cbox_dict)
n_vars = len(statevars)
if sediment_option:
    C = np.zeros((n_boxes, n_vars, n_iter + 1))
    CS1 = np.zeros((n_boxes, n_vars-2, n_iter + 1))
    CS2 = np.zeros((n_boxes, n_vars-2, n_iter + 1))
    
    for i, (box, values) in enumerate(initial_Cbox_dict.items()):
        for j, var in enumerate(statevars):
            C[i, j, 0] = values[var]
    for i, (box, values) in enumerate(initial_CS1box_dict.items()):
        for j, var in enumerate(statevars_Sed1):
            CS1[i, j, 0] = values[var]        
    for i, (box, values) in enumerate(initial_CS2box_dict.items()):
        for j, var in enumerate(statevars_Sed2):
            CS2[i, j, 0] = values[var] 
else:
    C = np.zeros((n_boxes, n_vars, n_iter + 1))
    for i, (box, values) in enumerate(initial_Cbox_dict.items()):
        for j, var in enumerate(statevars):
            C[i, j, 0] = values[var]

inflow_cols = np.array([[df_input_Q_box.columns.get_loc(col) for col in df_input_Q_box.columns if col.endswith(f"_To_{i + 1}")] for i in range(n_boxes)], dtype=object)
outflow_cols = np.array([[df_input_Q_box.columns.get_loc(col) for col in df_input_Q_box.columns if col.startswith(f"From_{i + 1}_")] for i in range(n_boxes)], dtype=object)

inflow_col_indices = np.array([np.array(cols, dtype=np.int64) for cols in inflow_cols], dtype=object)
outflow_col_indices = np.array([np.array(cols, dtype=np.int64) for cols in outflow_cols], dtype=object)

bc_input_C_array = np.array([[value[var] for var in statevars] for key, value in bc_input_C.items()])


input_Q_box_array = (df_input_Q_box.values * 86400).astype(np.float64)
flux_col_names = df_input_Q_box.columns.tolist()[1:]

T_boxes = T_boxes.iloc[:, 1:].to_numpy().T
V_boxes = V_boxes.iloc[:, 1:].to_numpy().T
Ia_boxes = Ia_boxes.iloc[:, 1:].to_numpy().T
Salt_boxes = Salt_boxes.iloc[:, 1:].to_numpy().T
fDay_boxes = fDay_boxes.iloc[:, :].to_numpy().T
H_boxes = np.array(list(H_boxes.values()), dtype=np.float64)
pmcb = np.array(list(pmcb.values()), dtype=np.float64)
smcb = np.array(list(smcb.values()), dtype=np.float64)

from_box_indices = np.array([int(col_name.split('_')[1]) for col_name in flux_col_names], dtype=np.int64)

inflow_col_indices = [np.array(cols, dtype=np.int64) for cols in inflow_cols]
outflow_col_indices = [np.array(cols, dtype=np.int64) for cols in outflow_cols]

if sediment_option:
    C = np.array(C, dtype=np.float64)
    CS1 = np.array(CS1, dtype=np.float64)
    CS2 = np.array(CS2, dtype=np.float64)
else:
    C = np.array(C, dtype=np.float64)
    
input_Q_box_array = np.array(input_Q_box_array, dtype=np.float64)
bc_input_C_array = np.array(bc_input_C_array, dtype=np.float64)

# ================================= fojit / ================================= #
# =========================================================================== #

# =========================================================================== #
# ======================= Numerical Solution for C \ ======================== #

@jit(cache=cache_option, nopython=True, parallel=parallel_option)
def flux(C, input_Q_box_array, bc_input_C_array, V, box_idx, time_step, dt, inflow_col_indices, outflow_col_indices, from_box_indices):
    V_box = V[box_idx][time_step - 1]
    current_concentration = C[box_idx, :, time_step - 1]  # Get all state variable concentrations for the box

    # Calculate inflow contributions
    inflow_sum = np.zeros(n_vars, dtype=np.float64)  # Array to hold inflow contributions for each variable
    for col_idx in inflow_col_indices[box_idx]:
        flux_value = input_Q_box_array[time_step - 1, col_idx]  # Access by index
        from_box_idx = from_box_indices[col_idx - 1]  # Use preprocessed index

        # Determine "from" concentrations based on the inflow source index
        if from_box_idx < 0:  # Boundary condition source
            source_concentration = bc_input_C_array[abs(from_box_idx) - 1, :, time_step - 1]  # All variables at once
        else:
            source_concentration = C[from_box_idx - 1, :, time_step - 1]  # All variables from another box

        inflow_sum += (flux_value / V_box) * source_concentration  # Update inflow for all variables

    # Calculate outflow contributions
    outflow_sum = np.zeros(n_vars, dtype=np.float64)
    for col_idx in outflow_col_indices[box_idx]:
        flux_value = input_Q_box_array[time_step - 1, col_idx]  # Access by index
        outflow_sum += (flux_value / V_box) * current_concentration  # Update outflow for all variables

    # Update concentrations for the box and variables
    C[box_idx, :, time_step] += (inflow_sum - outflow_sum) * dt

@jit(cache=cache_option, nopython=True, parallel=parallel_option)
def simulate_Cp(
        C, 
        input_Q_box_array, bc_input_C_array, 
        V_boxes, T_boxes, Ia_boxes, Salt_boxes, fDay_boxes, H_boxes, 
        pmcb, 
        dt, 
        inflow_col_indices, outflow_col_indices, from_box_indices, 
        ):
        
    for t in range(1, n_iter + 1):
        for box_idx in prange(n_boxes):
            T_box = T_boxes[box_idx, t - 1]
            V_box = V_boxes[box_idx, t - 1]
            H_box = H_boxes[box_idx]
            Ia_box = Ia_boxes[box_idx, t - 1]
            Salt_box = Salt_boxes[box_idx, t - 1]
            fDay_box = fDay_boxes[box_idx, t - 1]
            kmc_box = pmcb[box_idx]

            box_conc = C[box_idx, :, t - 1].copy()
            
            R_t = pelagic_process_rates(box_conc, T_box, V_box, H_box, Ia_box, Salt_box, fDay_box, kmc_box)
            C[box_idx, :, t]   =   C[box_idx, :, t - 1] + R_t * dt
            
            flux(C, input_Q_box_array, bc_input_C_array, V_boxes, box_idx, t, dt, inflow_col_indices, outflow_col_indices, from_box_indices)
            
            C[box_idx, :, t] = np.maximum(C[box_idx, :, t], 1e-15)  # Discard negligibly small values

    return C

@jit(cache=cache_option, nopython=True, parallel=parallel_option)
def simulate_Cs(
        C, CS1, CS2, 
        input_Q_box_array, bc_input_C_array, 
        V_boxes, T_boxes, Ia_boxes, Salt_boxes, fDay_boxes, H_boxes, 
        pmcb, smcb, 
        dt, 
        inflow_col_indices, outflow_col_indices, from_box_indices, 
        ):
        
    for t in range(1, n_iter + 1):
        for box_idx in prange(n_boxes):
            T_box = T_boxes[box_idx, t - 1]
            V_box = V_boxes[box_idx, t - 1]
            H_box = H_boxes[box_idx]
            Ia_box = Ia_boxes[box_idx, t - 1]
            Salt_box = Salt_boxes[box_idx, t - 1]
            fDay_box = fDay_boxes[box_idx, t - 1]
            kmc_box = pmcb[box_idx]
            smc_box = smcb[box_idx]

            box_conc = C[box_idx, :, t - 1].copy()

            box_conc_S1 = CS1[box_idx, :, t - 1].copy()
            box_conc_S2 = CS2[box_idx, :, t - 1].copy()
            R_t, R_t_S1, R_t_S2 = pelagic_with_sediment_process_rates(box_conc, box_conc_S1, box_conc_S2, T_box, V_box, H_box, Ia_box, Salt_box, fDay_box, kmc_box, smc_box)
            C[box_idx, :, t]   =   C[box_idx, :, t - 1] + R_t    * dt
            CS1[box_idx, :, t] = CS1[box_idx, :, t - 1] + R_t_S1 * dt
            CS2[box_idx, :, t] = CS2[box_idx, :, t - 1] + R_t_S2 * dt
            
            flux(C, input_Q_box_array, bc_input_C_array, V_boxes, box_idx, t, dt, inflow_col_indices, outflow_col_indices, from_box_indices)
            
            C[box_idx, :, t] = np.maximum(C[box_idx, :, t], 1e-15)  # Discard negligibly small values

    return C


# ======================= Numerical Solution for C / ======================== #
# =========================================================================== #



# =========================================================================== #
# ============================== Run Eutropy \ ============================== #

# Run the simulation according to sediment_option
if sediment_option:
    Cb = simulate_Cs(C, CS1, CS2, 
                    input_Q_box_array, bc_input_C_array,
                    V_boxes, T_boxes, Ia_boxes, Salt_boxes, fDay_boxes, H_boxes, 
                    pmcb, smcb, 
                    dt, 
                    inflow_col_indices, outflow_col_indices, 
                    from_box_indices
                    )
else:
    Cb = simulate_Cp(C,
                    input_Q_box_array, bc_input_C_array,
                    V_boxes, T_boxes, Ia_boxes, Salt_boxes, fDay_boxes, H_boxes, 
                    pmcb,
                    dt, 
                    inflow_col_indices, outflow_col_indices, 
                    from_box_indices
                    )    


# Save simulation output for the specified box(es)
for box in box_out: 
    C_dict = convert_C_to_dict(Cb[box-1], stateVars_toPlot)
    save_C_to_csv(C_dict, sim_start_date, dt, n_iter, f"boxOut_{box}.csv")


# =========================================================================== #
# =========================================================================== #
# Save simulation outputs for non-intrusive optimization tools.
# Works only observations introduced in the config.py file
# R2, RE, PBIAS and simulation outputs corresponding to observation dates.

if ifObs:
    
    for box in box_obs:     
        opt_out.write_out(f'outputs/boxOut_{box}.csv', 
                          f'observations/{box}_calibration.csv', 
                          calib_sDate)
        opt_out.write_out(f'outputs/boxOut_{box}.csv', 
                          f'observations/{box}_validation.csv', 
                          valid_sDate)

# =========================================================================== #
# =========================================================================== #

elapsed_time = time.time() - start_time
print('\nsimulation took: %.2f seconds\n' % elapsed_time)
print('# =================================================================== #')
# =============================== Run CuLPy / =============================== #
# =========================================================================== #
