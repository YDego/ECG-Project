import wfdb
import numpy as np
import processing_functions as pre_processing


def get_records(dataset):
    record_path = dataset['path'] + '\RECORDS'
    with open(record_path, 'r') as f:
        records = f.readlines()
    return records, len(records)


def select_lead(dataset, selected_lead=None):

    if not selected_lead:
        leads = dataset['leads']

        # Get user input for which lead to plot
        lead = str(input(f"Choose a lead to plot ({', '.join(leads)}) [default ii]: ") or 'ii')

        if lead not in leads:
            raise ValueError("Invalid lead selection!")
    else:
        lead = selected_lead

    if dataset["name"] == "ludb":
        dataset["annotations"] = lead

    return lead


def get_dataset(selected_dataset_name=None):

    database = ['ludb', 'qt']

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

    if not selected_dataset_name:
        # Get user input for which dataset to plot
        dataset_name = str(input(f"Choose a dataset ({', '.join(database)}) [default ludb]: ") or 'ludb')
    else:
        dataset_name = selected_dataset_name

    if dataset_name == "qt":
        dataset = qt_dataset
    elif dataset_name == "ludb":
        dataset = ludb_dataset
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

    lead = select_lead(dataset, selected_lead)

    # Get a single signal from the records
    ecg_signal = ecg_record.__dict__['p_signal'][:, dataset["leads"].index(lead)]
    annotation = wfdb.rdann(data_path, dataset["annotations"])
    ann_markers = annotation.symbol
    annotation_sample = np.ndarray.tolist(annotation.sample)
    fs = ecg_record.fs

    ecg_dict = {
        "dataset": dataset['name'],
        "record": ecg_record,
        "signal": ecg_signal,
        "name": record_name,
        "ann": annotation_sample,
        "ann_markers": ann_markers,
        "lead": lead,
        "fs": fs
    }

    return ecg_dict


if __name__ == "__main__":

    # Call the function with leads and file_count as inputs
    ecg_dict = ecg_lead_ext()

    # ECG processing
    ecg_processed_signal = pre_processing.ecg_pre_processing(ecg_dict)
