# import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftfreq, ifft, fftshift, ifftshift
import pywt
# from wfdb import processing
# from scipy.signal import butter
# from scipy import signal


def baseline_removal_moving_median(signal, fs):
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
    window_size = 2 * fs
    filtered_signal = signal - np.convolve(signal, np.ones(window_size) / window_size, mode='same')
    filtered_signal = filtered_signal - np.convolve(filtered_signal, np.ones(window_size) / window_size, mode='same')
    return filtered_signal


def low_pass_filter(cutoff_freq, signal, sampling_rate):
    spectrum = fftshift(fft(signal))
    freq = fftshift(fftfreq(signal.shape[-1], 1 / sampling_rate))
    spectrum[0:(round((len(spectrum) * (sampling_rate / 2 - cutoff_freq)) / sampling_rate) + 1)] = 0
    spectrum[(signal.shape[-1] - round((len(spectrum) * (sampling_rate / 2 - cutoff_freq)) / sampling_rate)): (
        signal.shape[-1])] = 0
    filter_signal = ifft(ifftshift(spectrum))
    return filter_signal, freq, spectrum


def high_pass_filter(cutoff_freq, signal, sampling_rate):
    freq = fftshift(fftfreq(signal.shape[-1], 1 / sampling_rate))
    spectrum_without_shift = (fft(signal))
    spectrum_without_shift[0:(round((len(spectrum_without_shift) * cutoff_freq) / sampling_rate) + 1)] = 0
    spectrum_without_shift[
        (signal.shape[-1] - round((len(spectrum_without_shift) * cutoff_freq) / sampling_rate)): (signal.shape[-1])] = 0
    filter_signal = ifft(spectrum_without_shift)
    return filter_signal, freq, fftshift(spectrum_without_shift)


def band_pass_filter(cutoff_freq_down, cutoff_freq_up, signal, sampling_rate):
    # freq = fftshift(fftfreq(signal.shape[-1], 1 / sampling_rate))
    spectrum = fftshift(fft(signal))
    spectrum[0:(round((len(spectrum) * (sampling_rate / 2 - cutoff_freq_up)) / sampling_rate) + 1)] = 0
    spectrum[(signal.shape[-1] - round((len(spectrum) * (sampling_rate / 2 - cutoff_freq_up)) / sampling_rate)): (
        signal.shape[-1])] = 0
    spectrum_without_shift = (ifftshift(spectrum))
    spectrum_without_shift[0:(round((len(spectrum_without_shift) * cutoff_freq_down) / sampling_rate) + 1)] = 0
    spectrum_without_shift[
        (signal.shape[-1] - round((len(spectrum_without_shift) * cutoff_freq_down) / sampling_rate)): (
            signal.shape[-1])] = 0
    filter_signal = np.real(ifft(spectrum_without_shift))
    return filter_signal


def compute_fft(signal, sample_rate):
    N = len(signal)
    fft_signal = np.abs(fft(signal-np.mean(signal))[0:N // 2])
    frequency_bins = fftfreq(N, 1/sample_rate)[:N // 2]

    return fft_signal, frequency_bins


def wavelet_filter(signal):
    wavelet = pywt.Wavelet('db2')
    # levdec = min(pywt.dwt_max_level(signal.shape[-1], wavelet.dec_len), 6)
    Ca4, Cd4, Cd3, Cd2, Cd1 = pywt.wavedec(signal, wavelet=wavelet, level=4)
    Ca4, Cd2, Cd1 = np.zeros(Ca4.shape[-1]), np.zeros(Cd2.shape[-1]), np.zeros(Cd1.shape[-1])
    filtered_signal = pywt.waverec([Ca4, Cd4, Cd3, Cd2, Cd1], wavelet)
    return filtered_signal


def ecg_pre_processing(ecg_dict):
    fs = ecg_dict['fs']
    ecg_processed = ecg_dict.copy()
    processed_signal = ecg_processed['signal']

    # Baseline removal
    processed_signal = baseline_removal_moving_median(processed_signal, fs)

    """
    if input("Perform powerline filter [y/N]? ") == "y":
        # Remove powerline interference
        powerline = [50, 60]
        bandwidth = 1
        ecg_filtered['signal'] = notch_filter(ecg_filtered['signal'], powerline, bandwidth, fs)

    if input("Perform BP filter [y/N]? ") == "y":
        # Remove high frequency noise
        ecg_filtered['signal'] = band_pass_filter(0.5, 50, ecg_filtered['signal'], fs)

    if input("Perform Wavelet filter [y/N]? ") == "y":
        # Remove high frequency noise
        ecg_filtered['signal'] = wavelet_filter(ecg_filtered['signal'])
    """
    ecg_processed['signal'] = processed_signal
    ecg_processed['fft'], ecg_processed['frequency_bins'] = compute_fft(ecg_processed["signal"], fs)

    return ecg_processed
