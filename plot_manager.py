import matplotlib.pyplot as plt
import numpy as np


def marker_converter(ann_markers):
    converter = {
        '(': '<',
        ')': '>',
        'p': 'X',
        'N': 'X',
        't': 'X'
    }
    markers_converted = ann_markers[:]
    for i in range(len(ann_markers)):
        markers_converted[i] = converter[ann_markers[i]]

    return markers_converted


def color_coverter(ann_markers):
    color_converter = {
        '(': 'k',
        ')': 'k',
        'p': 'b',
        'N': 'g',
        't': 'r'
    }
    color_converted = ann_markers[:]
    for i in range(len(ann_markers)):
        color_converted[i] = color_converter[ann_markers[i]]

    return color_converted


def plot_single_signal(ecg_dict):
    fft = ecg_dict["fft"]
    frequency_bins = ecg_dict["frequency_bins"]

    # Calculate time array
    time = [i / ecg_dict['fs'] for i in range(len(ecg_dict['original_signal']))] ####### 16.6 change to ecg_dict['original_signal'] from ecg_dict['signal']

    # Plot the signal
    plt.subplot(2, 1, 1)
    plt.plot(time, ecg_dict['original_signal'])#######
    plt.title(f'ECG Lead {ecg_dict["lead"]} for datafile {ecg_dict["name"]}')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (mV)')
    if ecg_dict['ann'] is not None:
        ann = ecg_dict['ann']
        markers = marker_converter(ecg_dict['ann_markers'])
        colors = color_coverter(ecg_dict['ann_markers'])
        j = 0

        for i in ann:
            plt.scatter(time[i], ecg_dict['original_signal'][i], c=colors[j], marker=markers[j])
            j += 1

    if ecg_dict['our_ann'] is not None:
        our_ann = ecg_dict['our_ann']
        markers = marker_converter(ecg_dict['our_ann_markers'].copy()) ## todo ask why with copy its not working
        colors = color_coverter(ecg_dict['our_ann_markers'])
        j = 0

        for i in our_ann:
            plt.scatter(time[i], ecg_dict['original_signal'][i], c=colors[j], marker=markers[j])
            j += 1


    # Plot the FFT
    plt.subplot(2, 1, 2)
    plt.plot(frequency_bins, np.abs(fft), color='red')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.title('FFT')

    # Display the plot
    plt.tight_layout()
    plt.show()


def plot_original_vs_processed(signal1, signal2, ann=False):
    fft1 = signal1["fft"]
    fft2 = signal2["fft"]
    freq_bin1 = signal1["frequency_bins"]
    freq_bin2 = signal2["frequency_bins"]

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    fs = signal1['fs']
    time = [i / fs for i in range(len(signal1['signal']))]

    # Plot the signal
    axs[0, 0].plot(time, signal1['signal'])
    axs[0, 0].set_ylabel('Amplitude (mV)')
    axs[0, 0].set_title('Original ECG Signal')
    if ann:
        axs[0, 0].scatter([time[i] for i in signal1['ann']], [signal1['signal'][i] for i in signal1['ann']], c='r')

    # Plot the FFT
    axs[1, 0].plot(freq_bin1, np.abs(fft1))
    axs[1, 0].set_xlabel('Frequency')
    axs[1, 0].set_ylabel('Magnitude')
    axs[1, 0].set_title('FFT')

    # Plot the signal
    axs[0, 1].plot(time, signal2['signal'], color='red')
    axs[0, 1].set_ylabel('Amplitude (mV)')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_title('Processed ECG Signal')
    if ann:
        axs[1].scatter([time[i] for i in signal2['ann']], [signal2['signal'][i] for i in signal2['ann']], c='r')

    # Plot the FFT
    axs[1, 1].plot(freq_bin2, np.abs(fft2), color='red')
    axs[1, 1].set_xlabel('Frequency')
    axs[1, 1].set_ylabel('Magnitude')
    axs[1, 1].set_title('FFT')

    axs[0, 0].sharex(axs[0, 1])
    axs[0, 0].sharey(axs[0, 1])
    axs[1, 0].sharex(axs[1, 1])
    axs[1, 0].sharey(axs[1, 1])

    plt.show()
