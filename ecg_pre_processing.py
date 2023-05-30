import numpy as np
import matplotlib.pyplot as plt
from wfdb import processing
from scipy.signal import butter
from scipy import signal


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def notch_filter(signal_data, sample_rate, freq_list):
    nyquist_rate = sample_rate / 2.0
    filtered_signal = signal_data.copy()

    for freq in freq_list:
        notch_width = 5.0  # in Hz
        notch_freq = freq / nyquist_rate
        b, a = signal.iirnotch(notch_freq, notch_width, sample_rate)
        filtered_signal = signal.filtfilt(b, a, filtered_signal)

    return filtered_signal


def baseline_removal_moving_median(signal, window_size=201):
    """
    Perform baseline removal using a moving median.

    Parameters:
    -----------
    signal : numpy array
        The signal to be filtered.
    window_size : int
        The size of the window for the moving median filter.

    Returns:
    --------
    filtered_signal : numpy array
        The baseline-corrected signal.
    """
    filtered_signal = signal - np.convolve(signal, np.ones(window_size)/window_size, mode='same')
    filtered_signal = filtered_signal - np.convolve(filtered_signal, np.ones(window_size)/window_size, mode='same')
    return filtered_signal


def ecg_pre_processing(ecg_record, ecg_signal):

    fs = ecg_record.fs
    ecg_filtered_signal = ecg_signal

    if input("Perform normalization [y/N]? ") == "y":
        # Normalization
        ecg_filtered_signal = processing.normalize_bound(ecg_filtered_signal)

    if input("Perform baseline removal [y/N]? ") == "y":
        # Remove baseline - moving median
        window_size_sec = 1
        window_size = fs * window_size_sec
        ecg_filtered_signal = baseline_removal_moving_median(ecg_filtered_signal, window_size)

    if input("Perform powerline filter [y/N]? ") == "y":
        # Remove powerline interference
        powerline = [50, 60]
        ecg_filtered_signal = notch_filter(ecg_filtered_signal, fs, powerline)

    if input("Perform BP filter [y/N]? ") == "y":
        # Remove high frequency noise
        ecg_filtered_signal = butter_bandpass_filter(ecg_filtered_signal, 0.5, 35, fs=fs, order=4)

    return ecg_filtered_signal


def plot_ecg_signals(signal1, signal2, fs):
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    time = [i / fs for i in range(len(signal1))]

    axs[0].plot(time, signal1)
    axs[0].set_ylabel('Amplitude (mV)')
    axs[0].set_title('Original ECG Signal')

    axs[1].plot(time, signal2)
    axs[1].set_ylabel('Amplitude (mV)')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_title('Processed ECG Signal')

    plt.show()


def qrs_detection(ecg_signal, fs):
    # Apply QRS detection using the Pan-Tompkins algorithm
    qrs_inds = processing.qrs.xqrs_detect(sig=ecg_signal, fs=fs)
    # Plot the ECG signal and the detected QRS complexes
    plt.plot(ecg_signal)
    plt.scatter(qrs_inds, ecg_signal[qrs_inds], c='r')
    plt.title('ECG Signal - QRS Detection')
    plt.xlabel('Sample number')
    plt.ylabel('Voltage (mV)')
    plt.show()
