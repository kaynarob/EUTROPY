import pandas as pd
from datetime import datetime


def save_C_to_csv(CX, sim_start_date, dt, n_iter, output_file_name):
    # Generate dates
    start_date = datetime.strptime(sim_start_date, '%Y-%m-%d')
    dates = [start_date + pd.Timedelta(hours=i * dt * 24) for i in range(n_iter + 1)]
    
    # Prepare data for output, CX is a dictionary with variables as keys
    data = {'Date': dates}
    for var, values in CX.items():
        data[var] = values[:n_iter + 1]  # Adjust to the simulation range
    
    # Create a DataFrame and set date as index
    df = pd.DataFrame(data)
    df.set_index("Date", inplace=True)
    
    # Resample to daily means and save to CSV
    df = df.resample('D').mean()
    df.to_csv(f'outputs/{output_file_name}')
    print(f"Saved output to {output_file_name}")

# matrix or array to dictionary
def convert_C_to_dict(Carr, statevars):
    # Initialize an empty dictionary for the result
    C_dict = {}

    for var_idx, var in enumerate(statevars):

        # Add the box data to the main dictionary, key: by box index
        C_dict[var] = Carr[var_idx, :].tolist()

    return C_dict

