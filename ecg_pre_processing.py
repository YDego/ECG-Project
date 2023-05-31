import numpy as np
import matplotlib.pyplot as plt
from wfdb import processing
from scipy.signal import butter
from scipy import signal

import filters_by_fft


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

    if input("Perform baseline removal [y/N]? ") == "y":
        # Remove baseline - moving median
        window_size_sec = 1
        window_size = fs * window_size_sec
        ecg_filtered_signal = baseline_removal_moving_median(ecg_filtered_signal, window_size)

    if input("Perform powerline filter [y/N]? ") == "y":
        # Remove powerline interference
        powerline = [50, 60]
        ecg_filtered_signal = filters_by_fft.notch_filter(powerline, [0.1], ecg_filtered_signal, fs)

    if input("Perform BP filter [y/N]? ") == "y":
        # Remove high frequency noise
        ecg_filtered_signal = filters_by_fft.band_pass_filter(0.5, 50, ecg_filtered_signal, fs)

    return ecg_filtered_signal


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
