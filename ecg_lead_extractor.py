import wfdb
import numpy as np
import matplotlib.pyplot as plt
import ecg_pre_processing as pre_processing


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


def ecg_lead_ext(dataset_name):

    ludb_dataset = {
        "name": "ludb",
        "path": "lobachevsky-university-electrocardiography-database-1.0.1",
        "leads": ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6'],
        "annotations": ""
    }

    qt_dataset = {
        "name": "qt",
        "path": "qt-database-1.0.0",
        "leads": ['i', 'ii'],
        "annotations": "pu0"
    }

    if dataset_name == "qt":
        dataset = qt_dataset
    elif dataset_name == "ludb":
        dataset = ludb_dataset
    else:
        raise ValueError("Invalid dataset selection!")

    # Load the RECORDS file and get the number of records
    records_file = rf'ecg_dataset\{dataset["path"]}\RECORDS'
    records, num_records = get_records(records_file)
    data_file = int(
        input(f"There are {num_records} records available, choose one (1-{num_records}) [default 1]: ") or 1)

    # Check if the user input is valid
    if data_file > num_records or data_file < 1:
        raise ValueError("Invalid data selection!")

    # Get the selected record name and load the ECG record
    record_name = records[data_file - 1].strip()
    record_path = f'ecg_dataset/{dataset["path"]}/{record_name}'
    ecg_record = wfdb.rdrecord(record_path)

    # Get user input for which lead to plot
    lead = select_lead(dataset["leads"])

    if lead is None:
        raise ValueError("Invalid lead selection!")
    elif dataset_name == "ludb":
        dataset["annotations"] = lead

    # Get a single signal from the records
    ecg_signal = ecg_record.__dict__['p_signal'][:, dataset["leads"].index(lead)]
    annotation = wfdb.rdann(record_path, dataset["annotations"])
    annotation_sample = np.ndarray.tolist(annotation.sample)
    fs = ecg_record.fs

    # Calculate time array
    time = [i / fs for i in range(len(ecg_signal))]

    # Plot the signal
    plt.plot(time, ecg_signal)
    plt.title(f'ECG Lead {lead} for {record_name}')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (mV)')
    plt.scatter([time[i] for i in annotation_sample], [ecg_signal[i] for i in annotation_sample], c='r')
    plt.show()

    return ecg_record, ecg_signal, lead, fs


def choose_lead_from_dataset():

    # Data list
    database = ['ludb', 'qt']
    # Get user input for which dataset to plot
    dataset_name = str(input(f"Choose a dataset ({', '.join(database)}) [default ludb]: ") or 'ludb')

    if dataset_name in database:
        return ecg_lead_ext(dataset_name)
    else:
        raise ValueError("Invalid dataset selection!")


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

    # ECG processing
    ecg_processed_signal = pre_processing.ecg_pre_processing(ecg_record, ecg_signal)

    # Plot
    plot_ecg_signals(ecg_signal, ecg_processed_signal, fs)

