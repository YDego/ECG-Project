from IPython.display import display

#import matplotlib.pyplot as plt
#import numpy as np
#import os
#import shutil
#import posixpath
import wfdb

record = wfdb.rdrecord(r'ecg_dataset\lobachevsky-university-electrocardiography-database-1.0.1\data\57')
#record = wfdb.rdrecord('a103l', pn_dir='challenge-2015/training/')

signal = record.p_signal[:, 5]

# Plot signal
wfdb.plot_wfdb(record=record, title='ECG')
wfdb.plot_items(signal=signal, fs=record.fs, title='ECG Lead V6')
display(record.__dict__)
