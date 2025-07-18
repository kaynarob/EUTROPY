# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_all_variables(df_output, df_observation, plot_start_date, plot_end_date, resample_daily):
    # Filter datasets for the specified date range
    df_output = df_output[(df_output.index >= plot_start_date) & (df_output.index <= plot_end_date)]
    df_observation = df_observation[(df_observation.index >= plot_start_date) & (df_observation.index <= plot_end_date)]
    
    df_output['Cpy'] = df_output['Cpy']/50*1000
    
    # If resample_daily is True, resample the outputs daily
    if resample_daily:
        df_output = df_output.resample('D').mean()
    
    # Iterate through the columns in df_output
    for variable in df_output.columns:
        plt.figure(figsize=(12, 6))
        
        # Plotting the simulation values
        plt.plot(df_output.index, df_output[variable], label='Simulation', linestyle='-', color='red')
        
        if variable in df_observation.columns:
            # Plotting the observation values
            plt.scatter(df_observation.index, df_observation[variable], color='blue', label='Observation')
        
        # Formatting the plot
        
        if variable == "Cpy":
            variable = '$Chl-a$ (µg/l)'
            plt.ylabel(f'{variable}')
        else:
            plt.ylabel(f'{variable} (mg/l)')
        
        
        plt.legend()
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


# Load the datasets
os.chdir('../')
output_file_name="outputs/boxOut_19.csv"
plot_start_date = "2013-01-01"
plot_end_date = "2017-01-01"
df_output = pd.read_csv(output_file_name, parse_dates=[0], index_col=0)
df_observation = pd.read_csv('observations/19_observation.csv', parse_dates=[0], index_col=0)

# Call the function with user input
plot_all_variables(df_output, df_observation, plot_start_date, plot_end_date, resample_daily=False)

