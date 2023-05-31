import matplotlib.pyplot as plt
import numpy as np


def plot_single_signal(ecg_dict):
    fft = ecg_dict["fft"]
    frequency_bins = ecg_dict["frequency_bins"]

    # Calculate time array
    time = [i / ecg_dict['fs'] for i in range(len(ecg_dict['signal']))]

    # Plot the signal
    plt.subplot(2, 1, 1)
    plt.plot(time, ecg_dict['signal'])
    plt.title(f'ECG Lead {ecg_dict["lead"]} for datafile {ecg_dict["name"]}')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (mV)')
    if ecg_dict['ann'] is not None:
        plt.scatter([time[i] for i in ecg_dict['ann']], [ecg_dict['signal'][i] for i in ecg_dict['ann']], c='r')

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
    fft2 = signal1["fft"]
    freq_bin1 = signal1["frequency_bins"]
    freq_bin2 = signal1["frequency_bins"]

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    fs = signal1['fs']
    time = [i / fs for i in range(len(signal1['signal']))]

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
