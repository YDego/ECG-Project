import wfdb
import pandas as pd
import matplotlib.pyplot as plt
from wfdb import processing
from scipy.signal import butter, lfilter
from scipy import signal


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def notch_filter(signal_data, sample_rate, freq_list):
    nyquist_rate = sample_rate / 2.0
    filtered_signal = signal_data.copy()

    for freq in freq_list:
        notch_width = 5.0  # in Hz
        notch_freq = freq / nyquist_rate
        b, a = signal.iirnotch(notch_freq, notch_width, sample_rate)
        filtered_signal = signal.filtfilt(b, a, filtered_signal)

    return filtered_signal


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


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
        return
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
        return

    if lead == 'All' or lead == 'all':
        # Plot
        wfdb.plot_wfdb(record=ecg_record, title='ECG')
        return ecg_record

    # Get a single signal from the records
    ecg_signal = ecg_record.__dict__['p_signal'][:, leads.index(lead)]
    fs = ecg_record.fs

    # Plot
    title = f'ECG signal over time\nECG Lead: {lead}, Rhythm: {rhythms}, Age: {age},Sex: {sex}'
    wfdb.plot_items(signal=ecg_signal, fs=ecg_record.fs, title=title, time_units='seconds', sig_units=['mV'],
                    ylabel=['Voltage [mV]'])

    return ecg_record, ecg_signal, lead, fs


def ecg_lead_qt():
    # Load the RECORDS file and get the number of records
    records_file = 'ecg_dataset/qt-database-1.0.0/RECORDS'
    records, num_records = get_records(records_file)

    # Ask the user to choose a record
    data_file = int(input(f"There are {num_records} records available, choose one (1-{num_records}) [default 1]: ") or 1)
    if data_file < 1 or data_file > num_records:
        print(f"Invalid record number! Please choose a number between 1 and {num_records}.")
        return

    # Get the selected record name and load the ECG record
    record_name = records[data_file - 1].strip()
    record_path = f'ecg_dataset/qt-database-1.0.0/{record_name}'
    ecg_record = wfdb.rdrecord(record_path)

    # Define the list of leads available
    leads = ['i', 'ii']

    # Get user input for which lead to plot
    lead = select_lead(leads)

    if lead is None:
        return

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

    return ecg_record, ecg_signal, lead, fs


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
        return


def ecg_processing(ecg_record, ecg_signal):

    fs = ecg_record.fs
    ecg_filtered_signal = ecg_signal

    if input("Perform normalization [y/N]? ") == "y":
        # Normalization
        ecg_filtered_signal = processing.normalize_bound(ecg_filtered_signal)

    if input("Perform baseline filter [y/N]? ") == "y":
        # Remove baseline wander
        baseline = butter_lowpass_filter(ecg_filtered_signal, cutoff=3, fs=fs, order=5)
        ecg_filtered_signal -= baseline

    if input("Perform powerline filter [y/N]? ") == "y":
        # Remove powerline interference
        powerline = [50, 60]
        ecg_filtered_signal = notch_filter(ecg_filtered_signal, fs, powerline)

    if input("Perform HP filter [y/N]? ") == "y":
        # Remove high frequency noise
        ecg_filtered_signal = butter_bandpass_filter(ecg_filtered_signal, 0.5, 35, fs=fs, order=4)

    return ecg_filtered_signal


def plot_ecg_signals(signal1, signal2, fs):
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    t = [i / fs for i in range(len(signal1))]

    axs[0].plot(t, signal1)
    axs[0].set_ylabel('Amplitude (mV)')
    axs[0].set_title('Original ECG Signal')

    axs[1].plot(t, signal2)
    axs[1].set_ylabel('Amplitude (mV)')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_title('Processed ECG Signal')

    plt.show()


if __name__ == "__main__":

    # Call the function with leads and file_count as inputs
    ecg_record, ecg_signal, lead, fs = choose_lead_from_dataset()

    # Apply QRS detection using the Pan-Tompkins algorithm
    ecg_processed_signal = ecg_processing(ecg_record, ecg_signal)

    # Plot
    plot_ecg_signals(ecg_signal, ecg_processed_signal, fs)

