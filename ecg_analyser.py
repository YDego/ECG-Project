import wfdb
import pandas as pd
# from IPython.display import display
# import os

file_count = 200
data_file = str(input(f"There are {file_count} data files, choose one [default 1]: ") or '1')

# Check if the user input is valid
if int(data_file) > file_count or int(data_file) < 1:
    print("Invalid data selection!")
else:
    # Read the CSV file
    df = pd.read_csv(
        r'C:\Users\ydego\Documents\GitHub\ECG-Project\ecg_dataset\lobachevsky-university-electrocardiography-database-1.0.1\ludb.csv')
    # Get the details of the selected record
    rhythms = df.iloc[int(data_file) - 1]['Rhythms']
    sex = df.iloc[int(data_file) - 1]['Sex']
    age = df.iloc[int(data_file) - 1]['Age']

    # Chose data file
    record_path = fr'C:\Users\ydego\Documents\GitHub\ECG-Project\ecg_dataset\lobachevsky-university-electrocardiography-database-1.0.1\data\{data_file}'
    # Read the record
    record = wfdb.rdrecord(record_path)
    # record = wfdb.rdrecord('a103l', pn_dir='challenge-2015/training/')

    # Define the list of leads available
    leads = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'all']

    # Get user input for which lead to plot
    lead = str(input(f"Choose a lead to plot ({', '.join(leads)}) [default ii]: ") or 'ii')

    if lead not in leads:
        print("Invalid lead selection!")
    else:
        if lead == 'All' or lead == 'all':
            # Plot
            wfdb.plot_wfdb(record=record, title='ECG')
        else:
            # Get a single signal from the records
            signal = record.__dict__['p_signal'][:, leads.index(lead)]
            # Plot
            title = f'ECG signal over time\nECG Lead: {lead}, Rhythm: {rhythms}, Age: {age},Sex: {sex}'
            wfdb.plot_items(signal=signal, fs=record.fs, title=title, time_units='seconds', sig_units=['mV'], ylabel=['Voltage [mV]'])

        # display(record.__dict__)
