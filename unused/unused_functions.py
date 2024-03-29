import wfdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wfdb import processing


def ecg_lead_ludb():

    # Load the RECORDS file and get the number of records
    records_file = r'../ecg_dataset/lobachevsky-university-electrocardiography-database-1.0.1/RECORDS'
    records, num_records = get_records(records_file)
    data_file = int(input(f"There are {num_records} records available, choose one (1-{num_records}) [default 1]: ") or 1)

    # Check if the user input is valid
    if data_file > num_records or data_file < 1:
        print("Invalid data selection!")
        return
    # Read the CSV file
    df = pd.read_csv(
        r'../ecg_dataset/lobachevsky-university-electrocardiography-database-1.0.1/ludb.csv')
    # Get the details of the selected record
    rhythms = df.iloc[int(data_file) - 1]['Rhythms']
    sex = df.iloc[int(data_file) - 1]['Sex']
    age = df.iloc[int(data_file) - 1]['Age']

    # Choose data file
    record_path = fr'ecg_dataset\lobachevsky-university-electrocardiography-database-1.0.1\data\{data_file}'
    # Read the record
    ecg_record = wfdb.rdrecord(record_path)

    # record = wfdb.rdrecord('a103l', pn_dir='challenge-2015/training/')

    # Define the list of leads available
    leads = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'all']

    # Get user input for which lead to plot
    lead = select_lead(leads)

    if lead is None:
        return

    if lead == 'All' or lead == 'all':
        # Plot
        wfdb.plot_wfdb(record=ecg_record, title='ECG')
        return ecg_record

    # Get a single signal from the records
    ecg_signal = ecg_record.__dict__['p_signal'][:, leads.index(lead)]
    annotation = wfdb.rdann(record_path, lead)
    annotation_sample = np.ndarray.tolist(annotation.sample)
    fs = ecg_record.fs

    # Plot
    title = f'ECG signal over time\nECG Lead: {lead}, Rhythm: {rhythms}, Age: {age},Sex: {sex}'
    wfdb.plot_items(signal=ecg_signal, fs=ecg_record.fs, title=title, time_units='seconds', sig_units=['mV'],
                    ylabel=['Voltage [mV]'])

    # Plot ECG signal
    t = np.arange(ecg_signal.shape[0]) / fs
    fig, ax = plt.subplots()
    ax.plot(t, ecg_signal, lw=2)
    ax.set_title("ECG Lead " + lead.upper())
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")

    a = wfdb.io.rdheader('100', pn_dir='mitdb')
    # Plot QRS complex annotations
    # ax.plot(t, ecg_signal[annotation.sample/fs], 'rx')
    plt.scatter(t[annotation_sample], ecg_signal[annotation_sample], c='r')
    plt.show()

    return ecg_record, ecg_signal, lead, fs


def ecg_lead_qt():
    # Load the RECORDS file and get the number of records
    records_file = '../ecg_dataset/qt-database-1.0.0/RECORDS'
    records, num_records = get_records(records_file)

    # Ask the user to choose a record
    data_file = int(input(f"There are {num_records} records available, choose one (1-{num_records}) [default 1]: ") or 1)
    if data_file < 1 or data_file > num_records:
        print(f"Invalid record number! Please choose a number between 1 and {num_records}.")
        return

    # Get the selected record name and load the ECG record
    record_name = records[data_file - 1].strip()
    record_path = f'ecg_dataset/qt-database-1.0.0/data/{record_name}'
    ecg_record = wfdb.rdrecord(record_path)

    # Define the list of leads available
    leads = ['i', 'ii']

    # Get user input for which lead to plot
    lead = select_lead(leads)

    if lead is None:
        return

    # Get a single signal from the records
    ecg_signal = ecg_record.__dict__['p_signal'][:, leads.index(lead)]
    annotation = wfdb.rdann(record_path, 'pu0')
    annotation_sample = np.ndarray.tolist(annotation.sample)
    fs = ecg_record.fs

    # Calculate time array
    t = [i / fs for i in range(len(ecg_signal))]

    # Plot the signal
    plt.plot(t, ecg_signal)
    plt.title(f'ECG Lead II for {record_name}')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (mV)')
    plt.scatter([t[i] for i in annotation_sample], [ecg_signal[i] for i in annotation_sample], c='r')
    plt.show()

    return ecg_record, ecg_signal, lead, fs


def qrs_detection(ecg_signal, fs):
    # Apply QRS detection using the Pan-Tompkins algorithm
    qrs_indxs = processing.qrs.xqrs_detect(sig=ecg_signal, fs=fs)
    # Plot the ECG signal and the detected QRS complexes
    plt.plot(ecg_signal)
    plt.scatter(qrs_indxs, ecg_signal[qrs_indxs], c='r')
    plt.title('ECG Signal - QRS Detection')
    plt.xlabel('Sample number')
    plt.ylabel('Voltage (mV)')
    plt.show()


def notch_filter(signal, freq_list, bandwidth, sample_rate):
    # Perform FFT on the signal
    fft_signal = fft(signal)

    # Calculate the frequency bins
    n = len(signal)
    frequency_bins = np.fft.fftfreq(n, d=1 / sample_rate)

    # Find the indices of the frequencies within the specified range
    indices = []
    for freq in freq_list:
        indices.extend(np.where((frequency_bins >= freq - bandwidth / 2) & (frequency_bins <= freq + bandwidth / 2))[0])

    # Set the corresponding frequency components to zero
    fft_signal[indices] = 0

    # Perform inverse FFT to obtain the filtered signal
    filtered_signal = ifft(fft_signal)

    # Return the real part of the filtered signal
    return np.real(filtered_signal)


def plot_original_vs_processed(signal1, signal2, ann=False):
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    time = [i / signal1['fs'] for i in range(len(signal1['signal']))]

    axs[0].plot(time, signal1['signal'])
    axs[0].set_ylabel('Amplitude (mV)')
    axs[0].set_title('Original ECG Signal')
    if ann:
        axs[0].scatter([time[i] for i in signal1['ann']], [signal1['signal'][i] for i in signal1['ann']], c='r')

    axs[1].plot(time, signal2['signal'])
    axs[1].set_ylabel('Amplitude (mV)')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_title('Processed ECG Signal')
    if ann:
        axs[1].scatter([time[i] for i in signal2['ann']], [signal2['signal'][i] for i in signal2['ann']], c='r')

    plt.show()

# Normalization
ecg_filtered_signal = [1, 2]
ecg_filtered_signal = processing.normalize_bound(ecg_filtered_signal)
