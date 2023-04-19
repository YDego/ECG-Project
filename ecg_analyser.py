from IPython.display import display
import wfdb
import os


# Chose data file
data_dir = 'ecg_dataset/lobachevsky-university-electrocardiography-database-1.0.1/data'
file_count = len(os.listdir(data_dir))
data_file = str(input(f"There are {file_count} data files, choose one: "))

# Define the list of leads available
leads = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'All']

# Get user input for which lead to plot
lead = input(f"Choose a lead to plot ({', '.join(leads)}): ")

# Check if the user input is valid
if lead not in leads:
    print("Invalid lead selection!")
elif lead == 'All' or lead == 'all':
    # Read the record and plot the selected lead
    record_path = fr'ecg_dataset\lobachevsky-university-electrocardiography-database-1.0.1\data\{data_file}'
    record = wfdb.rdrecord(record_path)
    # record = wfdb.rdrecord('a103l', pn_dir='challenge-2015/training/')
    wfdb.plot_wfdb(record=record, title='ECG')
else:
    # Read the record and plot the selected lead
    record_path = fr'ecg_dataset\lobachevsky-university-electrocardiography-database-1.0.1\data\{data_file}'
    record = wfdb.rdrecord(record_path)
    # record = wfdb.rdrecord('a103l', pn_dir='challenge-2015/training/')
    signal = record.__dict__['p_signal'][:, leads.index(lead)]
    wfdb.plot_items(signal=signal, fs=record.fs, title=f'ECG Lead {lead}', time_units='seconds', sig_name='mV')

    display(record.__dict__)
