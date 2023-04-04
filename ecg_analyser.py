from IPython.display import display

#import matplotlib.pyplot as plt
#import numpy as np
#import os
#import shutil
#import posixpath

import wfdb

SAMPLE_RATE = 500 #[Hz]

record = wfdb.rdrecord(r'ecg_dataset\lobachevsky-university-electrocardiography-database-1.0.1\data\1')
#record = wfdb.rdrecord('a103l', pn_dir='challenge-2015/training/')

wfdb.plot_wfdb(record=record, title='ECG')
display(record.__dict__)