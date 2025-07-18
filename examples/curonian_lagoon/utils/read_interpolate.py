import pandas as pd
from datetime import datetime, timedelta


def kmc_reader(file_name):
    kmc_dict = {}
    f = open(file_name, "r+")
    for line in f:
        line_ = line.split()
        kmc_name = line_[0]
        kmc_values = float(line_[2])
        kmc_dict[kmc_name] = kmc_values
    f.close()
    return kmc_dict

def interpolate_wJDay(sim_start_date, sim_end_date, JDay_start_date, time_step_per_day, file_name):
    dt_minutes = 24 * 60 / time_step_per_day
    inputF_path = f'{file_name}.csv'
    inputF = pd.read_csv(inputF_path)
    start_date = datetime.strptime(JDay_start_date, "%Y-%m-%d")
    inputF['datetime'] = inputF['time'].apply(lambda x: start_date + timedelta(days=x))
    inputF.set_index('datetime', inplace=True)
    # Create new time index for interpolation
    new_time_index = pd.date_range(start=sim_start_date, end=sim_end_date, freq=f'{int(dt_minutes)}min')
    # Interpolate bc_flow data to new time index
    interpolated_input = inputF.reindex(inputF.index.union(new_time_index)).interpolate(method='time').loc[new_time_index]
    return interpolated_input

def interpolate_wDate(sim_start_date, sim_end_date, time_step_per_day, file_name):
    dt_minutes = 24 * 60 / time_step_per_day
    inputF_path = f'{file_name}.csv'
    inputF = pd.read_csv(inputF_path)
    inputF['time'] = pd.to_datetime(inputF['time'])
    inputF.set_index('time', inplace=True)
    new_time_index = pd.date_range(start=sim_start_date, end=sim_end_date, freq=f'{int(dt_minutes)}min')
    # Interpolate bc_concentration data to new time index
    if isinstance(inputF.index, pd.DatetimeIndex):
        interpolated_input = inputF.reindex(inputF.index.union(new_time_index)).interpolate(method='time').loc[new_time_index]
    else:
        raise ValueError("input does not have a DatetimeIndex, cannot perform time-based interpolation.")
    return interpolated_input


# specific order in the array of the model kinetic constants
kmc_keys = [
    'k_growth', 'k_resipration', 'k_mortality', 'k_excration', 'k_salt_death', 
    'v_set_Cpy', 'K_SN', 'K_SP', 'K_Sl_salt', 'K_Sl_ox_Cpy', 
    'k_c_decomp', 'k_n_decomp', 'k_p_decomp', 'v_set_Cpoc', 'v_set_Cpon', 
    'v_set_Cpop', 'k_c_mnr_ox', 'k_c_mnr_ni', 'k_n_mnr_ox', 'k_n_mnr_ni', 
    'k_p_mnr_ox', 'k_p_mnr_ni', 'K_Sl_Cpoc_decomp', 'K_Sl_Cpon_decomp', 
    'K_Sl_Cpop_decomp', 'K_Sl_c_mnr_ox', 'K_Sl_c_mnr_ni', 'K_Sl_ox_mnr_c', 
    'K_Si_ox_mnr_c', 'K_Sl_ni_mnr_c', 'K_Sl_n_mnr_ox', 'K_Sl_n_mnr_ni', 
    'K_Sl_ox_mnr_n', 'K_Si_ox_mnr_n', 'K_Sl_ni_mnr_n', 'K_Sl_p_mnr_ox', 
    'K_Sl_p_mnr_ni', 'K_Sl_ox_mnr_p', 'K_Si_ox_mnr_p', 'K_Sl_ni_mnr_p', 
    'k_nitrification', 'K_Sl_nitr', 'K_Sl_nitr_ox', 'k_denitrification', 
    'K_Sl_denitr', 'K_Si_denitr_ox', 'k_raer', 'K_be', 'I_s', 
    'theta_growth', 'theta_resipration', 'theta_mortality', 'theta_excration', 
    'theta_c_decomp', 'theta_n_decomp', 'theta_p_decomp', 'theta_c_mnr_ox', 
    'theta_n_mnr_ox', 'theta_p_mnr_ox', 'theta_c_mnr_ni', 'theta_n_mnr_ni', 
    'theta_p_mnr_ni', 'theta_nitr', 'theta_denitr', 'theta_rear', 
    'a_C_chl', 'a_N_C', 'a_P_C', 'a_O2_C', 'pabsorption'
    ]

smc_keys = [
    'omega_01_POC', 'omega_12_POC1', 'w_burial_POC1', 'k_dissol_POC1',                 
    'k_dissol_POC2', 'omega_01_PON', 'omega_12_PON1', 'w_burial_PON1',                 
    'k_dissol_PON1', 'k_dissol_PON2', 'omega_01_POP', 'omega_12_POP1',                 
    'w_burial_POP1', 'k_dissol_POP1', 'k_dissol_POP2', 'Kl_01_DOC',                     
    'Kl_12_DOC1', 'Kl_01_DON', 'Kl_12_DON1', 'Kl_01_DOP', 'Kl_12_DOP1',                    
    'Kl_01_NH4', 'Kl_12_NH4_1', 'k_nitr_NH4_1', 'Kl_01_NO3', 'Kl_12_NO3',                     
    'k_denitr_NO3_1', 'k_denitr_NO3_2', 'Kl_01_PO4', 'Kl_12_PO4_1',                   
    'K_HS_sed_POC_1_disl', 'K_HS_sed_POC_2_disl', 'K_HS_sed_PON_1_disl',           
    'K_HS_sed_PON_2_disl', 'K_HS_sed_POP_1_disl', 'K_HS_sed_POP_2_disl',           
    'K_HS_sed_Nitr', 'K_HS_sed_deNitr', 'k_DOC_1_Miner_dox',             
    'k_DON_1_Miner_dox', 'k_DOP_1_Miner_dox', 'K_HS_DOC_1_Minr_lim_dox',       
    'K_HS_DON_1_Minr_lim_dox', 'K_HS_DOP_1_Minr_lim_dox', 'K_HS_DOC_1_Minr_dox',           
    'K_HS_DON_1_Minr_dox', 'K_HS_DOP_1_Minr_dox', 'k_DOC_1_Miner_no3_1',           
    'K_HS_DOC_1_Minr_inhb_dox_1', 'K_HS_DOC_1_Minr_lim_no3_1', 'K_HS_DOC_1_Minr_no3_1',         
    'k_DON_1_Miner_no3_1', 'K_HS_DON_1_Minr_inhb_dox_1', 'K_HS_DON_1_Minr_lim_no3_1',     
    'K_HS_DON_1_Minr_no3_1', 'k_DOP_1_Miner_no3_1', 'K_HS_DOP_1_Minr_inhb_dox_1',    
    'K_HS_DOP_1_Minr_lim_no3_1', 'K_HS_DOP_1_Minr_no3_1', 'k_adsorp_PO4',                  
    'K_HS_adsorp_PO4'
    ]