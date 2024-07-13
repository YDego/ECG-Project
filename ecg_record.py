import wfdb


class ECGRecord:
    def __init__(self, dataset, record_name):
        self.dataset = dataset
        self.record_name = record_name
        self.data_path = f"{dataset.path}\\data\\{record_name}"
        self.record = wfdb.rdrecord(self.data_path)
        self.fs = self.record.fs
        self.ecg_signal = None
        self.annotations = None

    def load_signal(self, lead):
        self.ecg_signal = self.record.__dict__['p_signal'][:, self.dataset.leads.index(lead)]
        self.annotations = wfdb.rdann(self.data_path, self.dataset.annotations)
