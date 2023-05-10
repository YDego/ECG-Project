import wfdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy
from wfdb import processing


def ecg_lead_ludb():
    # Load the RECORDS file and get the number of records
    records_file = r'ecg_dataset\lobachevsky-university-electrocardiography-database-1.0.1\RECORDS'
    with open(records_file, 'r') as f:
        records = f.readlines()
    num_records = len(records)
    data_file = int(input(f"There are {num_records} records available, choose one (1-{num_records}) [default 1]: ") or 1)
    signal = numpy.empty(0)
    record = None

    # Check if the user input is valid
    if data_file > num_records or data_file < 1:
        print("Invalid data selection!")
        return record, signal
    # Read the CSV file
    df = pd.read_csv(
        r'ecg_dataset\lobachevsky-university-electrocardiography-database-1.0.1\ludb.csv')
    # Get the details of the selected record
    rhythms = df.iloc[int(data_file) - 1]['Rhythms']
    sex = df.iloc[int(data_file) - 1]['Sex']
    age = df.iloc[int(data_file) - 1]['Age']

    # Chose data file
    record_path = fr'ecg_dataset\lobachevsky-university-electrocardiography-database-1.0.1\data\{data_file}'
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
            wfdb.plot_items(signal=signal, fs=record.fs, title=title, time_units='seconds', sig_units=['mV'],
                            ylabel=['Voltage [mV]'])
        # display(record.__dict__)
    return record, signal


def ecg_lead_qt():
    signal = numpy.empty(0)
    record = None
    # Load the RECORDS file and get the number of records
    records_file = 'ecg_dataset/qt-database-1.0.0/RECORDS'
    with open(records_file, 'r') as f:
        records = f.readlines()
    num_records = len(records)

    # Ask the user to choose a record
    data_file = int(input(f"There are {num_records} records available, choose one (1-{num_records}) [default 1]: ") or 1)
    if data_file < 1 or data_file > num_records:
        print(f"Invalid record number! Please choose a number between 1 and {num_records}.")
        return record, signal

    # Get the selected record name and load the ECG record
    record_name = records[data_file - 1].strip()
    record_path = f'ecg_dataset/qt-database-1.0.0/{record_name}'
    record = wfdb.rdrecord(record_path)

    # Define the list of leads available
    leads = ['i', 'ii']

    # Get user input for which lead to plot
    lead = str(input(f"Choose a lead to plot ({', '.join(leads)}) [default ii]: ") or 'ii')

    if lead not in leads:
        print("Invalid lead selection!")
    else:
        # Get a single signal from the records
        signal = record.__dict__['p_signal'][:, leads.index(lead)]
        # Plot the signal
        plt.plot(signal)
        plt.title(f'ECG Lead I - {record_name}')
        plt.xlabel('Sample Number')
        plt.ylabel('Voltage (mV)')
        plt.show()

    return record, signal


def choose_lead_from_dataset():

    # Data list
    database = ['ludb', 'qt']
    # Get user input for which dataset to plot
    dataset = str(input(f"Choose a dataset ({', '.join(database)}) [default ludb]: ") or 'ludb')

    if dataset == 'ludb':
        return ecg_lead_ludb()
    elif dataset == 'qt':
        return ecg_lead_qt()
    else:
        print("Invalid dataset selection!")
        return None, numpy.empty(0)


if __name__ == "__main__":

    # Call the function with leads and file_count as inputs
    record, signal = choose_lead_from_dataset()
    # Apply QRS detection using the Pan-Tompkins algorithm
    if record is not None:
        qrs_inds = processing.qrs.gqrs_detect(sig=signal, fs=record.fs)
        # Plot the ECG signal and the detected QRS complexes
        plt.plot(signal)
        plt.title('Processed data')
        plt.scatter(qrs_inds, signal[qrs_inds], c='r')
        plt.xlabel('Sample number')
        plt.ylabel('Voltage (mV)')
        plt.show()
