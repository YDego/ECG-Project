# import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftfreq, ifft, fftshift, ifftshift
import pywt
import qrs_detection
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
    freq = fftshift(fftfreq(signal.shape[-1], 1 / sampling_rate))
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


def notch_filter(signal, freq_list, bandwidth, sample_rate):
    # Perform FFT on the signal
    fft_signal = fft(signal)

    # Calculate the frequency bins
    n = len(signal)
    frequency_bins = np.fft.fftfreq(n, d=1 / sample_rate)

    # Find the indices of the frequencies within the specified range
    indices = []
    for freq in freq_list:
        indices.extend(np.where((frequency_bins >= freq - bandwidth / 2) & (frequency_bins <= freq + bandwidth / 2))[0])

    # Set the corresponding frequency components to zero
    fft_signal[indices] = 0

    # Perform inverse FFT to obtain the filtered signal
    filtered_signal = ifft(fft_signal)

    # Return the real part of the filtered signal
    return np.real(filtered_signal)


def compute_fft(signal, sample_rate):
    N = len(signal)
    fft_signal = np.abs(fft(signal-np.mean(signal))[0:N // 2])
    frequency_bins = fftfreq(N, 1/sample_rate)[:N // 2]

    return fft_signal, frequency_bins


def wavelet_filter(signal):
    wavelet = pywt.Wavelet('sym4')
    # levdec = min(pywt.dwt_max_level(signal.shape[-1], wavelet.dec_len), 6)
    Ca4, Cd4, Cd3, Cd2, Cd1 = pywt.wavedec(signal, wavelet=wavelet, level=4)
    Ca4, Cd2, Cd1 = np.zeros(Ca4.shape[-1]), np.zeros(Cd2.shape[-1]), np.zeros(Cd1.shape[-1])
    filtered_signal = pywt.waverec([Ca4, Cd4, Cd3, Cd2, Cd1], wavelet)
    return filtered_signal


def ecg_pre_processing(ecg_dict):
    fs = ecg_dict['fs']
    ecg_filtered = ecg_dict.copy()

    #if input("Perform QRS detection [y/N]? ") == "y":
    ecg_filtered = qrs_detection.detection_qrs(ecg_filtered)

    #if input("Perform comparison between our annotations and real annotations [y/N]? ") == "y":
    ecg_filtered = qrs_detection.comprasion_r_peaks(ecg_filtered)

    """
    if input("Perform baseline removal [y/N]? ") == "y":
        # Remove baseline - moving median
        window_size_sec = 1
        window_size = fs * window_size_sec
        ecg_filtered['signal'] = baseline_removal_moving_median(ecg_filtered['signal'], window_size)

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
    ecg_filtered["fft"], ecg_filtered["frequency_bins"] = compute_fft(ecg_filtered["signal"], ecg_filtered["fs"])

    return ecg_filtered
