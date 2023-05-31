# import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftfreq, ifft, fftshift, ifftshift
# from wfdb import processing
# from scipy.signal import butter
# from scipy import signal


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


def low_pass_filter(cutoff_freq, signal, sampling_rate):
    spectrum = fftshift(fft(signal))
    freq = fftshift(fftfreq(signal.shape[-1], 1 / sampling_rate))
    spectrum[0:(round((len(spectrum) * (sampling_rate / 2 - cutoff_freq)) / sampling_rate) + 1)] = 0
    spectrum[(signal.shape[-1] - round((len(spectrum) * (sampling_rate / 2 - cutoff_freq)) / sampling_rate)): (signal.shape[-1])] = 0
    filter_signal = ifft(ifftshift(spectrum))
    return filter_signal, freq, spectrum


def high_pass_filter(cutoff_freq, signal, sampling_rate):
    freq = fftshift(fftfreq(signal.shape[-1], 1 / sampling_rate))
    spectrum_without_shift = (fft(signal))
    spectrum_without_shift[0:(round((len(spectrum_without_shift) * cutoff_freq) / sampling_rate) + 1)] = 0
    spectrum_without_shift[(signal.shape[-1] - round((len(spectrum_without_shift) * cutoff_freq) / sampling_rate)): (signal.shape[-1])] = 0
    filter_signal = ifft(spectrum_without_shift)
    return filter_signal, freq, fftshift(spectrum_without_shift)


def band_pass_filter(cutoff_freq_down, cutoff_freq_up, signal, sampling_rate):
    freq = fftshift(fftfreq(signal.shape[-1], 1 / sampling_rate))
    spectrum = fftshift(fft(signal))
    spectrum[0:(round((len(spectrum) * (sampling_rate / 2 - cutoff_freq_up)) / sampling_rate) + 1)] = 0
    spectrum[(signal.shape[-1] - round((len(spectrum) * (sampling_rate / 2 - cutoff_freq_up)) / sampling_rate)): (signal.shape[-1])] = 0
    spectrum_without_shift = (ifftshift(spectrum))
    spectrum_without_shift[0:(round((len(spectrum_without_shift) * cutoff_freq_down) / sampling_rate) + 1)] = 0
    spectrum_without_shift[(signal.shape[-1] - round((len(spectrum_without_shift) * cutoff_freq_down) / sampling_rate)): (signal.shape[-1])] = 0
    filter_signal = ifft(spectrum_without_shift)
    return filter_signal


def notch_filter(frequencies, bandwidth, signal, sampling_rate):
    filtered_signal = signal[:]
    for center_freq in frequencies:
        cutoff_freq_down = center_freq - bandwidth / 2
        cutoff_freq_up = center_freq + bandwidth / 2
        filtered_signal = band_pass_filter(cutoff_freq_down, cutoff_freq_up, filtered_signal, sampling_rate)
    return filtered_signal


def ecg_pre_processing(ecg_dict):

    fs = ecg_dict['fs']
    ecg_filtered = {
        "dataset": ecg_dict["dataset"],
        "record": ecg_dict["record"],
        "signal": ecg_dict["signal"],
        "name": ecg_dict["name"],
        "ann": ecg_dict["ann"],
        "lead": ecg_dict["lead"],
        "fs": ecg_dict["fs"]
    }

    if input("Perform baseline removal [y/N]? ") == "y":
        # Remove baseline - moving median
        window_size_sec = 1
        window_size = fs * window_size_sec
        ecg_filtered['signal'] = baseline_removal_moving_median(ecg_filtered['signal'], window_size)

    if input("Perform powerline filter [y/N]? ") == "y":
        # Remove powerline interference
        powerline = [50, 60]
        ecg_filtered['signal'] = notch_filter(powerline, 2, ecg_filtered['signal'], fs)

    if input("Perform BP filter [y/N]? ") == "y":
        # Remove high frequency noise
        ecg_filtered['signal'] = band_pass_filter(0.5, 50, ecg_filtered['signal'], fs)

    return ecg_filtered
