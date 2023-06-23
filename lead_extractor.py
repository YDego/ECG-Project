import wfdb
import numpy as np
import processing_functions as pf



def get_records(dataset):
    record_path = dataset['path'] + '\RECORDS'
    with open(record_path, 'r') as f:
        records = f.readlines()
    return records, len(records)


def select_lead(dataset, selected_lead=None, data_path=''):

    if not selected_lead:
        leads = []
        if dataset["name"] == 'mit':
            with open(data_path + '.hea') as f:
                lines = f.readlines()
                leads = [lines[1].split()[-1], lines[2].split()[-1]]
            dataset['leads'] = leads
        else:
            leads = dataset['leads']

        # Get user input for which lead to plot
        lead = str(input(f"Choose a lead to plot ({', '.join(leads)}) [default first lead]: ") or leads[0])

        if lead not in leads:
            raise ValueError("Invalid lead selection!")
    else:
        lead = selected_lead

    if dataset["name"] == "ludb":
        dataset["annotations"] = lead

    return lead


def get_dataset(selected_dataset_name=None):

    database = ['ludb', 'qt', 'mit']

    ludb_dataset = {
        "name": "ludb",
        "path": "ecg_dataset\lobachevsky-university-electrocardiography-database-1.0.1",
        "leads": ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6'],
        "annotations": "",
    }

    qt_dataset = {
        "name": "qt",
        "path": "ecg_dataset\qt-database-1.0.0",
        "leads": ['i', 'ii'],
        "annotations": "pu0"
    }

    mit_dataset = {
        "name": "mit",
        "path": "ecg_dataset\mit-bih-arrhythmia-database-1.0.0",
        "leads": [],
        "annotations": "atr"
    }

    if not selected_dataset_name:
        # Get user input for which dataset to plot
        dataset_name = str(input(f"Choose a dataset ({', '.join(database)}) [default ludb]: ") or 'ludb')
    else:
        dataset_name = selected_dataset_name

    if dataset_name == "qt":
        dataset = qt_dataset
    elif dataset_name == "ludb":
        dataset = ludb_dataset
    elif dataset_name == "mit":
        dataset = mit_dataset
    else:
        raise ValueError("Invalid dataset selection!")

    return dataset


def get_datafile(num_records):
    data_file = int(
        input(f"There are {num_records} records available, choose one (1-{num_records}) [default 1]: ") or 1)

    # Check if the user input is valid
    if data_file > num_records or data_file < 1:
        raise ValueError("Invalid data selection!")

    return data_file


def ecg_lead_ext(selected_dataset=None, selected_data_file=None, selected_lead=None):

    dataset = get_dataset(selected_dataset)

    # Load the RECORDS file and get the number of records
    records, num_records = get_records(dataset)

    if not selected_data_file:
        # Get the data file
        data_file = get_datafile(num_records)
    else:
        data_file = selected_data_file

    # Get the selected record name and load the ECG record
    record_name = records[data_file - 1].strip().replace('data/', '')
    data_path = dataset['path'] + f'\\data\\{record_name}'
    ecg_record = wfdb.rdrecord(data_path)

    lead = select_lead(dataset, selected_lead, data_path=data_path)

    # Get a single signal from the records
    ecg_signal = ecg_record.__dict__['p_signal'][:, dataset["leads"].index(lead)]
    annotation = wfdb.rdann(data_path, dataset["annotations"])
    ann_markers = annotation.symbol
    annotation_sample = np.ndarray.tolist(annotation.sample)
    fs = ecg_record.fs

    # Cut signals
    signal_len = 10  # [sec]
    ecg_signal = ecg_signal[1:signal_len * fs]
    cut_index = np.argmin(np.abs(np.array(annotation_sample)-(signal_len * fs)))
    annotation_sample = annotation_sample[1:cut_index]
    ann_markers = ann_markers[1:cut_index]
    if dataset['name'] == 'mit':
        ann_markers = np.full(len(ann_markers), 'N').tolist()
    baseline_removal_signal = pf.baseline_removal_moving_median(ecg_signal, fs * 1)




    baseline_removal_signal = pf.baseline_removal_moving_median(ecg_signal, fs * 1)
    # FFT
    fft, frequency_bins = pf.compute_fft(ecg_signal, fs)

    ecg_dict = {
        "dataset": dataset['name'],
        "record": ecg_record,
        "original_signal" : baseline_removal_signal, ## changed 23.6 01:24
        "signal": baseline_removal_signal,
        "name": record_name,
        "ann": annotation_sample,
        "ann_markers": ann_markers,
        "our_ann": [], ## samples
        "our_ann_markers": [], ## strings
        "r_peak_success": [0, 0],
        "lead": lead,
        "fs": fs,
        "fft": fft,
        "frequency_bins": frequency_bins
    }

    return ecg_dict


if __name__ == "__main__":

    # Call the function with leads and file_count as inputs
    ecg_dict = ecg_lead_ext()

    # ECG processing
    ecg_processed_signal = pf.ecg_pre_processing(ecg_dict)
