import time
import math
import wfdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pywt
import filters_by_fft
import qrs_detection
import p_t_detection
from wfdb import processing
import scipy

# from IPython.display import display
# import os
start_time = time.time()
file_count = 200
data_file = '1'
list_of_distance = []
while data_file != '201':
    # Check if the user input is valid
    #data_file = str(input(f"There are {file_count} data files, choose one [default 1]: ") or '1')

    #print(data_file)
    if int(data_file) > file_count or int(data_file) < 1:
        print("Invalid data selection!")
    else:
        # Read the CSV file
        df = pd.read_csv(r'../ecg_dataset/lobachevsky-university-electrocardiography-database-1.0.1/ludb.csv')
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
       # lead = str(input(f"Choose a lead to plot ({', '.join(leads)}) [default ii]: ") or 'ii')
        lead = 'ii'

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
                annotation = wfdb.rdann(record_path, lead)
                annotation_sample = np.ndarray.tolist(annotation.sample)

                title = f'ECG signal over time\nECG Lead: {lead}, Rhythm: {rhythms}, Age: {age},Sex: {sex}'
                #wfdb.plot_items(signal=signal, fs=record.fs, title=title, time_units='seconds', sig_units=['mV'], ylabel=['Voltage [mV]'])
                ## Apply QRS detection using the Pan-Tompkins algorithm
                ###qrs_inds = processing.qrs.xqrs_detect(sig=signal, fs=record.fs)
                # local_peaks = wfdb.processing.find_local_peaks(sig=signal, radius=25)

                original_signal = signal
                #original_signal, freq, spectrum = filters_by_fft.band_pass_filter(0.5,100, signal, record.fs)
                #plt.plot(original_signal)
                #plt.show()
                #plt.plot(np.arange(0,5000), np.real(band_pass_filter_signal))
                #plt.show()



                ###################try to use in dwt for denoising and then find the right R picks
                #wavelet = pywt.Wavelet('sym3')
                ####levdec = min(pywt.dwt_max_level(signal.shape[-1], wavelet.dec_len), 6)
                #Ca4, Cd4, Cd3, Cd2, Cd1 = pywt.wavedec(signal, wavelet=wavelet, level=4)

                #Ca4, Cd2, Cd1 = np.zeros(Ca4.shape[-1]),np.zeros(Cd2.shape[-1]),np.zeros(Cd1.shape[-1])
                #new_signal = pywt.waverec([Ca4, Cd4, Cd3, Cd2, Cd1], wavelet)

                #plt.plot(new_signal)
                #plt.show()
                plt.xlabel('Sample number')
                plt.ylabel('Voltage (mV)')

                #wavelet = pywt.Wavelet('gaus1')
                #print(wavelet)
                ######################using cwt for R peaks
                #scales = np.arange(1,31,1)
                #print(scipy.signal.daub(4))
                #coef, freqs = pywt.cwt(original_signal, scales, 'db3')## Daubechie 3
                ###################ploting scalogram

                #plt.figure(figsize=(15,10))
                #plt.imshow(abs(coef),extent=[0,5000,100,1],interpolation='bilinear',cmap='bone' , aspect='auto',vmax=abs(coef).max(), vmin=-abs(coef).max())
                #plt.gca().invert_yaxis()
                #plt.xticks(np.arange(0,5000,1000))
                #plt.yticks(np.arange(1,101,10))
                #plt.show()


                # Plot the ECG signal and the detected QRS complexes
                #plt.plot(signal)

                #plt.xlabel('Sample number')
                #plt.ylabel('Voltage (mV)')
                #plt.show()

                new_signal, freq, spectrum = filters_by_fft.band_pass_filter(8, 50, signal, record.fs)
                plt.plot(original_signal)
                q_and_s_annotation = qrs_detection.annotation_for_q_and_s(annotation_sample)
                plt.scatter(q_and_s_annotation, original_signal[q_and_s_annotation], c='k')#
                #plt.scatter(annotation_sample,original_signal[annotation_sample],c='k')
                threshold = np.mean(abs(new_signal)**3)
                open_dots, R_peaks, closed_dots, all_dots = qrs_detection.detect_qrs(original_signal, abs(new_signal) ** 3, threshold)#
                plt.scatter(R_peaks, original_signal[R_peaks], c='b')
                #plt.scatter(local_peaks,original_signal[local_peaks], c='k')
                distance_from_real = qrs_detection.distance_from_real_dot(q_and_s_annotation, all_dots)
                print(distance_from_real)
                plt.plot(abs(new_signal)**3)#
                plt.scatter(open_dots, original_signal[open_dots], c='r')
                plt.scatter(closed_dots, original_signal[closed_dots], c='g')
                plt.xlabel('Sample number')
                plt.ylabel('Voltage (mV)')
                plt.show()

                data_file = str(int(data_file)+1)
                list_of_distance.append(distance_from_real)





                #plt.scatter(R_peaks, signal[R_peaks], c='b')
                #plt.plot(freq,np.real(spectrum))
                #plt.xlabel('Frequency[Hz]')
                #plt.ylabel('Amplitude')
                #plt.show()



                #high_pass_filter_signal,  freq, spectrum = filters_by_fft.high_pass_filter(5,signal,record.fs)
                #plt.plot(np.arange(0, 5000), np.real(high_pass_filter_signal))
                #plt.show()

                #print(record.__dict__)

mask = np.zeros(len(list_of_distance), dtype=bool)
for index, x in enumerate(list_of_distance):
    new_x = abs(x)
    flag = True
    for value in new_x:
        if value >= 20:
            flag = False
    mask[index] = flag
good = 0
for index, x in enumerate(mask):
    if x == True:
        good = good + 1
    if x == False:
        print(index+1)
print(f"There are {good} good and {200-good} bad ")

