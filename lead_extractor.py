import wfdb
import numpy as np
import processing_functions as pre_processing


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
    records_path = rf'ecg_dataset\{dataset["path"]}\RECORDS'
    records, num_records = get_records(records_path)
    data_file = int(
        input(f"There are {num_records} records available, choose one (1-{num_records}) [default 1]: ") or 1)

    # Check if the user input is valid
    if data_file > num_records or data_file < 1:
        raise ValueError("Invalid data selection!")

    # Get the selected record name and load the ECG record
    record_name = records[data_file - 1].strip().replace('data/', '')
    data_path = f'ecg_dataset/{dataset["path"]}/data/{record_name}'
    ecg_record = wfdb.rdrecord(data_path)

    # Get user input for which lead to plot
    lead = select_lead(dataset["leads"])

    if lead is None:
        raise ValueError("Invalid lead selection!")
    elif dataset_name == "ludb":
        dataset["annotations"] = lead

    # Get a single signal from the records
    ecg_signal = ecg_record.__dict__['p_signal'][:, dataset["leads"].index(lead)]
    annotation = wfdb.rdann(data_path, dataset["annotations"])
    ann_markers = annotation.symbol
    annotation_sample = np.ndarray.tolist(annotation.sample)
    fs = ecg_record.fs

    ecg_dict = {
        "dataset": dataset_name,
        "record": ecg_record,
        "signal": ecg_signal,
        "name": record_name,
        "ann": annotation_sample,
        "ann_markers": ann_markers,
        "lead": lead,
        "fs": fs
    }

    return ecg_dict


def choose_lead_from_dataset():

    # Data list
    database = ['ludb', 'qt']
    # Get user input for which dataset to plot
    dataset_name = str(input(f"Choose a dataset ({', '.join(database)}) [default ludb]: ") or 'ludb')

    if dataset_name in database:
        return ecg_lead_ext(dataset_name)
    else:
        raise ValueError("Invalid dataset selection!")


if __name__ == "__main__":

    # Call the function with leads and file_count as inputs
    ecg_dict = choose_lead_from_dataset()

    # ECG processing
    ecg_processed_signal = pre_processing.ecg_pre_processing(ecg_dict['record'], ecg_dict['signal'])
