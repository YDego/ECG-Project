import wfdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy
from wfdb import processing


def get_records(records_file):
    with open(records_file, 'r') as f:
        records = f.readlines()
    return records, len(records)


def select_lead(leads):
    # Get user input for which lead to plot
    lead = str(input(f"Choose a lead to plot ({', '.join(leads)}) [default ii]: ") or 'ii')

    if lead not in leads:
        print("Invalid lead selection!")
        return None

    return lead


def ecg_lead_ludb():
    # Load the RECORDS file and get the number of records
    records_file = r'ecg_dataset\lobachevsky-university-electrocardiography-database-1.0.1\RECORDS'
    records, num_records = get_records(records_file)
    data_file = int(input(f"There are {num_records} records available, choose one (1-{num_records}) [default 1]: ") or 1)

    # Check if the user input is valid
    if data_file > num_records or data_file < 1:
        print("Invalid data selection!")
        return None, numpy.empty(0)
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
    ecg_record = wfdb.rdrecord(record_path)
    # record = wfdb.rdrecord('a103l', pn_dir='challenge-2015/training/')

    # Define the list of leads available
    leads = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'all']

    # Get user input for which lead to plot
    lead = select_lead(leads)

    if lead is None:
        return None, None

    if lead == 'All' or lead == 'all':
        # Plot
        wfdb.plot_wfdb(record=ecg_record, title='ECG')
        return ecg_record, None

    # Get a single signal from the records
    ecg_signal = ecg_record.__dict__['p_signal'][:, leads.index(lead)]
    # Plot
    title = f'ECG signal over time\nECG Lead: {lead}, Rhythm: {rhythms}, Age: {age},Sex: {sex}'
    wfdb.plot_items(signal=ecg_signal, fs=ecg_record.fs, title=title, time_units='seconds', sig_units=['mV'],
                    ylabel=['Voltage [mV]'])

    return ecg_record, ecg_signal


def ecg_lead_qt():
    # Load the RECORDS file and get the number of records
    records_file = 'ecg_dataset/qt-database-1.0.0/RECORDS'
    records, num_records = get_records(records_file)

    # Ask the user to choose a record
    data_file = int(input(f"There are {num_records} records available, choose one (1-{num_records}) [default 1]: ") or 1)
    if data_file < 1 or data_file > num_records:
        print(f"Invalid record number! Please choose a number between 1 and {num_records}.")
        return None, None

    # Get the selected record name and load the ECG record
    record_name = records[data_file - 1].strip()
    record_path = f'ecg_dataset/qt-database-1.0.0/{record_name}'
    ecg_record = wfdb.rdrecord(record_path)

    # Define the list of leads available
    leads = ['i', 'ii']

    # Get user input for which lead to plot
    lead = select_lead(leads)

    if lead is None:
        return None, None

    # Get a single signal from the records
    ecg_signal = ecg_record.__dict__['p_signal'][:, leads.index(lead)]
    fs = ecg_record.fs

    # Calculate time array
    times = [i / fs for i in range(len(ecg_signal))]

    # Plot the signal
    plt.plot(times, ecg_signal)
    plt.title(f'ECG Lead II for {record_name}')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (mV)')
    plt.show()

    return ecg_record, ecg_signal


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


def ecg_processing(ecg_record, ecg_signal):

    # Apply QRS detection using the Pan-Tompkins algorithm
    if ecg_record is not None:
        qrs_inds = processing.qrs.gqrs_detect(sig=ecg_signal, fs=ecg_record.fs)
        # Plot the ECG signal and the detected QRS complexes
        plt.plot(ecg_signal)
        plt.title('Processed data')
        plt.scatter(qrs_inds, ecg_signal[qrs_inds], c='r')
        plt.xlabel('Sample number')
        plt.ylabel('Voltage (mV)')
        plt.show()


if __name__ == "__main__":

    # Call the function with leads and file_count as inputs
    ecg_record, ecg_signal = choose_lead_from_dataset()

    # Apply QRS detection using the Pan-Tompkins algorithm
    ecg_processing(ecg_record, ecg_signal)
