import wfdb


class ECGDataset:
    def __init__(self, name, path, leads, annotations):
        self.name = name
        self.path = path
        self.leads = leads
        self.annotations = annotations

    @staticmethod
    def load_dataset(dataset_name, annotation=None):
        datasets = {
            'ludb': ECGDataset("ludb", "ecg_dataset\\lobachevsky-university-electrocardiography-database-1.0.1",
                               ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6'], ""),
            'qt': ECGDataset("qt", "ecg_dataset\\qt-database-1.0.0", ['i', 'ii'], annotation or "pu1"),
            'mit': ECGDataset("mit", "ecg_dataset\\mit-bih-arrhythmia-database-1.0.0", [], "atr")
        }
        return datasets.get(dataset_name)

    def get_records(self):
        record_path = f"{self.path}\\RECORDS"
        with open(record_path, 'r') as f:
            records = f.readlines()
        return records, len(records)

    def select_lead(self, selected_lead=None, data_path=''):
        if not selected_lead:
            leads = self.leads if self.name != 'mit' else self._get_mit_leads(data_path)
            self.leads = leads
            lead = leads[0] if self.name != 'mit' else self._select_mit_lead(leads)
        else:
            lead = selected_lead

        if self.name == "ludb":
            self.annotations = lead
        return lead

    def _get_mit_leads(self, data_path):
        with open(data_path + '.hea') as f:
            lines = f.readlines()
            leads = [lines[1].split()[-1], lines[2].split()[-1]]
        return leads

    def _select_mit_lead(self, leads):
        if 'MLII' in leads:
            return 'MLII'
        elif 'MLI' in leads:
            return 'MLI'
        else:
            return 'V5'
