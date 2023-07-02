import matplotlib.pyplot as plt
import numpy as np


def marker_converter(ann_markers):
    marker_dict = {
        '(': '<',
        ')': '>',
        'p': 'X',
        'N': 'X',
        'n': 'o',
        't': 'X'
    }
    markers_converted = ann_markers[:]
    for i in range(len(ann_markers)):
        markers_converted[i] = marker_dict[ann_markers[i]]

    return markers_converted


def color_converter(ann_markers):
    color_dict = {
        '(': 'k',
        ')': 'k',
        'p': 'b',
        'N': 'g',
        'n': 'g',
        't': 'r'
    }
    colors = ann_markers[:]
    for i in range(len(ann_markers)):
        colors[i] = color_dict[ann_markers[i]]

    return colors


def plot_ann(ann, ann_markers, signal, time, plotter):
    markers = marker_converter(ann_markers.copy())
    colors = color_converter(ann_markers)
    j = 0

    for i in ann:
        plotter.scatter(time[i], signal[i], c=colors[j], marker=markers[j])
        j += 1


def plot_single_signal(ecg_dict, ann=False, our_ann=False):
    fft = ecg_dict["fft"]
    frequency_bins = ecg_dict["frequency_bins"]
    signal = ecg_dict['original_signal']
    # Calculate time array
    time = [i / ecg_dict['fs'] for i in range(len(ecg_dict['original_signal']))]  # 16.6 change to ecg_dict['original_signal'] from ecg_dict['signal']

    # Plot the signal
    plt.subplot(2, 1, 1)
    plt.plot(time, signal)
    plt.title(f'Database: {ecg_dict["dataset"]}, datafile: {ecg_dict["name"]}, Lead {ecg_dict["lead"]}')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (mV)')
    if ann:
        plot_ann(ecg_dict['ann'], ecg_dict['ann_markers'], signal, time, plt)

    if our_ann:
        plot_ann(ecg_dict['our_ann'], ecg_dict['our_ann_markers'], signal, time, plt)

    # Plot the FFT
    plt.subplot(2, 1, 2)
    plt.plot(frequency_bins, np.abs(fft), color='red')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.title('FFT')

    # Display the plot
    plt.tight_layout()
    plt.show()


def plot_original_vs_processed(ecg_dict_1, ecg_dict_2, ann=False, our_ann=False):
    signal1 = ecg_dict_1["signal"]
    signal2 = ecg_dict_2["signal"]
    fft1 = ecg_dict_1["fft"]
    fft2 = ecg_dict_2["fft"]
    freq_bin1 = ecg_dict_1["frequency_bins"]
    freq_bin2 = ecg_dict_2["frequency_bins"]

    fig, axs = plt.subplots(2, 2)
    fs = ecg_dict_1['fs']
    time = [i / fs for i in range(len(signal1))]

    # Plot the signal
    axs[0, 0].plot(time, signal1)
    axs[0, 0].set_ylabel('Amplitude (mV)')
    axs[0, 0].set_title('Original ECG Signal')
    if ann:
        plot_ann(ecg_dict_1['ann'], ecg_dict_1['ann_markers'], signal1, time, axs[0, 0])

    # Plot the FFT
    axs[1, 0].plot(freq_bin1, np.abs(fft1))
    axs[1, 0].set_xlabel('Frequency')
    axs[1, 0].set_ylabel('Magnitude')
    axs[1, 0].set_title('FFT')

    # Plot the signal
    axs[0, 1].plot(time, signal2, color='red')
    axs[0, 1].set_ylabel('Amplitude (mV)')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_title('Processed ECG Signal')
    if ann:
        plot_ann(ecg_dict_2['ann'], ecg_dict_2['ann_markers'], signal2, time, axs[0, 1])
    if our_ann:
        plot_ann(ecg_dict_2['our_ann'], ecg_dict_2['our_ann_markers'], signal2, time, axs[0, 1])

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
