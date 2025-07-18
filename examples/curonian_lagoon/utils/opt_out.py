# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import scipy
import os


def relative_error(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred) / np.sum(y_true))

def r2_score(y_true, y_pred):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_true, y_pred)
    return r_value**2

def pbias(y_true, y_pred):
    return (np.sum(y_true - y_pred)/np.sum(y_true))*100

def write_out(*args):
    # Extract the start date from the last argument
    start_date = args[-1]

    # The rest of the arguments are pairs of simulation and observation files
    file_pairs = args[:-1]

    # Ensure it has pairs of files (simulation and observation)
    if len(file_pairs) % 2 != 0:
        raise ValueError("You must provide pairs of simulation and observation files.")

    # Open the 'opt.out' file in 'w' mode to overwrite any previous results
    with open('outputs/opt.out', 'w') as opt_file:
        # Process each simulation and observation pair
        for i in range(0, len(file_pairs), 2):
            simulation_file = file_pairs[i]
            observation_file = file_pairs[i + 1]

            # Load the data
            simulation_df = pd.read_csv(simulation_file)
            observation_df = pd.read_csv(observation_file)
            simulation_df['Cpy'] = simulation_df['Cpy']/50*1000

            # Correct the 'date' column format for both dataframes
            simulation_df['Date'] = pd.to_datetime(simulation_df['Date'])
            observation_df['Date'] = pd.to_datetime(observation_df['Date'], format='%m/%d/%Y')

            # Resample the simulation output to daily frequency
            simulation_daily_df = simulation_df.set_index('Date').resample('D').mean().reset_index()

            # Filter data starting from the selected start date
            simulation_daily_df = simulation_daily_df[simulation_daily_df['Date'] >= pd.to_datetime(start_date)]
            observation_df = observation_df[observation_df['Date'] >= pd.to_datetime(start_date)]

            # Align the simulation output with the observation data
            aligned_df = pd.merge(simulation_daily_df, observation_df, on='Date', suffixes=('_sim', '_obs'))

            # Extract base names from the files (without directory path or extensions)
            observation_base_name = os.path.splitext(os.path.basename(observation_file))[0]

            # Create filenames for outputs based on the observation file name
            r2_file_name = f"outputs/r2_values_{observation_base_name}.txt"
            re_file_name = f"outputs/re_values_{observation_base_name}.txt"
            pbias_file_name = f"outputs/pbias_values_{observation_base_name}.txt"

            # Calculate R2, RE and PBIAS values for each column
            r2_values = {}
            re_values = {}
            pbias_values = {}

            columns = ['Cpy']
            for column in columns:
                sim_column = column + '_sim'
                obs_column = column + '_obs'
                
                # Drop rows where either the observation or simulation data has NaN values
                valid_idx = ~aligned_df[[sim_column, obs_column]].isnull().any(axis=1)
                
                # Calculate statistics only for valid data points (non-NaN in both columns)
                valid_sim = aligned_df.loc[valid_idx, sim_column]
                valid_obs = aligned_df.loc[valid_idx, obs_column]
                
                if not valid_sim.empty and not valid_obs.empty:
                    r2_values[column] = r2_score(valid_obs, valid_sim)
                    re_values[column] = relative_error(valid_obs, valid_sim)
                    pbias_values[column] = pbias(valid_obs, valid_sim)
                    
                    # Write valid simulation values to the opt.out file
                    for value in valid_sim:
                        opt_file.write(f'{value}\n')

            # Write R2 values to a text file
            with open(r2_file_name, 'w') as file:
                for column, value in r2_values.items():
                    file.write(f'{column}: {value}\n')

            # Write relative error (RE) values to a text file
            with open(re_file_name, 'w') as file:
                for column, value in re_values.items():
                    file.write(f'{column}: {value}\n')

            # Write PBIAS values to a text file
            with open(pbias_file_name, 'w') as file:
                for column, value in pbias_values.items():
                    file.write(f'{column}: {value}\n')

