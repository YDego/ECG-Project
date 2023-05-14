import math

import wfdb
from wfdb import processing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pywt
import filters_by_fft

# from IPython.display import display
# import os

file_count = 200
data_file = 'start'
while data_file != 'exit':
    # Check if the user input is valid
    data_file = str(input(f"There are {file_count} data files, choose one [default 1]: ") or '1')
    if int(data_file) > file_count or int(data_file) < 1:
        print("Invalid data selection!")
    else:
        # Read the CSV file
        df = pd.read_csv(r'ecg_dataset\lobachevsky-university-electrocardiography-database-1.0.1\ludb.csv')
        # Get the details of the selected record
        rhythms = df.iloc[int(data_file) - 1]['Rhythms']
        sex = df.iloc[int(data_file) - 1]['Sex']
        age = df.iloc[int(data_file) - 1]['Age']

        # Chose data file
        record_path = fr'ecg_dataset\lobachevsky-university-electrocardiography-database-1.0.1\data\{data_file}'
        # Read the record
        record = wfdb.rdrecord(record_path)
        # record = wfdb.rdrecord('a103l', pn_dir='challenge-2015/training/')

        # Define the list of leads available
        leads = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'all']

        # Get user input for which lead to plot
        lead = str(input(f"Choose a lead to plot ({', '.join(leads)}) [default ii]: ") or 'ii')

        if lead not in leads:
            print("Invalid lead selection!")
        else:
            if lead == 'All' or lead == 'all':
                # Plot
                wfdb.plot_wfdb(record=record, title='ECG')
            else:
                # Get a single signal from the records
                signal = record.__dict__['p_signal'][:, leads.index(lead)]
                ##print(signal)
                # Plot
                title = f'ECG signal over time\nECG Lead: {lead}, Rhythm: {rhythms}, Age: {age},Sex: {sex}'
                ##wfdb.plot_items(signal=signal, fs=record.fs, title=title, time_units='seconds', sig_units=['mV'], ylabel=['Voltage [mV]'])
                ## Apply QRS detection using the Pan-Tompkins algorithm
                ###qrs_inds = processing.qrs.xqrs_detect(sig=signal, fs=record.fs)
                #local_peaks = wfdb.processing.find_local_peaks(sig=signal, radius=25)


                ###################try to use in dwt for denoising and then find the right R picks
                #wavelet = pywt.Wavelet('sym4')
                ####levdec = min(pywt.dwt_max_level(signal.shape[-1], wavelet.dec_len), 6)
                #Ca4,Cd4,Cd3,Cd2,Cd1  = pywt.wavedec(signal, wavelet=wavelet, level=4)

                #Ca4,Cd2,Cd1 = np.zeros(Ca4.shape[-1]),np.zeros(Cd2.shape[-1]),np.zeros(Cd1.shape[-1])
                #new_signal = pywt.waverec([Ca4,Cd4,Cd3,Cd2,Cd1], wavelet)

                ###plt.plot(new_signal)
                #plt.plot(abs(new_signal)**2)
                #plt.xlabel('Sample number')
                #plt.ylabel('Voltage (mV)')
                #plt.show()

                ######################using cwt for R peaks
                #scales = np.arange(1,101)
                #coef,freqs = pywt.cwt(signal, scales, 'gaus1')
                ###################ploting scalogram

                #plt.figure(figsize=(15,10))
                #plt.imshow(abs(coef),extent=[0,5000,100,1],interpolation='bilinear',cmap='bone' , aspect='auto',vmax=abs(coef).max(), vmin=-abs(coef).max())
                #plt.gca().invert_yaxis()
                #plt.xticks(np.arange(0,5000,1000))
                #plt.yticks(np.arange(1,101,10))
                #plt.show()

                #R_peaks = wfdb.processing.find_local_peaks(signal, radius=500)


                # Plot the ECG signal and the detected QRS complexes
                #plt.plot(signal)
                #plt.scatter(R_peaks, signal[R_peaks], c='b')
                #plt.xlabel('Sample number')
                #plt.ylabel('Voltage (mV)')
                #plt.show()


                plt.plot(signal)
                #plt.scatter(local_peaks, signal[local_peaks], c='r')
                plt.xlabel('Sample number')
                plt.ylabel('Voltage (mV)')
                plt.show()







                # plt.scatter(R_peaks, signal[R_peaks], c='b')
                #plt.plot(freq,np.real(spectrum))
                #plt.xlabel('Frequency[Hz]')
                #plt.ylabel('Amplitude')
                #plt.show(

            band_pass_filter_signal , freq , spectrum = filters_by_fft.band_pass_filter(5,30,signal,record.fs)
            plt.plot(np.arange(0,5000) ,np.real(band_pass_filter_signal))
            plt.show()

            #print(record.__dict__)


