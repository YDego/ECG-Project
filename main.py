from ecg_dataset import ECGDataset
from ecg_record import ECGRecord
from ecg_processor import ECGProcessor
import processing_functions as pf


def main():
    selected_dataset = input("Choose a dataset (ludb, qt, mit) [default ludb]: ") or 'ludb'
    dataset = ECGDataset.load_dataset(selected_dataset)

    records, num_records = dataset.get_records()
    data_file = int(input(f"There are {num_records} records available, choose one (1-{num_records}) [default 1]: ") or 1)
    record_name = records[data_file - 1].strip().replace('data/', '')

    ecg_record = ECGRecord(dataset, record_name)
    lead = dataset.select_lead()
    ecg_record.load_signal(lead)

    processor = ECGProcessor(ecg_record, signal_len=10)
    ecg_dict = processor.process()

    # Plot the original and processed signals with annotations
    processor.plot_first_segment()

    # ECG processing
    ecg_processed_signal = pf.ecg_pre_processing(ecg_dict)
    print("ECG processing complete.")


if __name__ == "__main__":
    main()
