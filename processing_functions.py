import numpy as np
from scipy.fftpack import fft, fftfreq, ifft, fftshift, ifftshift
from scipy.signal import find_peaks


def baseline_removal_moving_median(signal, fs, window_size_in_sec=0.2):
    window_size = int(window_size_in_sec * fs)  # 200ms window size
    median = np.median(signal)
    baseline = np.convolve(signal, np.ones(window_size) / window_size, mode='same')
    return signal - baseline + median


def band_pass_filter(cutoff_freq_down, cutoff_freq_up, signal, sampling_rate):
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


def compute_fft(signal, fs):
    freqs = np.fft.fftfreq(len(signal), 1/fs)
    fft_vals = np.abs(np.fft.fft(signal))
    return fft_vals, freqs


def qrs_detection(ecg_signal, fs):
    # Apply band-pass filter
    filtered_signal = band_pass_filter(8, 49, ecg_signal, fs)

    # Enhance the signal by squaring
    processed_signal = filtered_signal ** 3

    # threshold
    threshold = np.mean(processed_signal[round(0.1*fs): processed_signal.shape[-1] - round(0.1*fs)])

    # Adjust the signal ends
    processed_signal = adjust_signal_ends(processed_signal, fs)

    # Dynamic thresholding
    threshold = np.mean(processed_signal)
    qrs_indices = find_peaks_above_threshold(processed_signal, threshold)

    return qrs_indices


def adjust_signal_ends(signal, fs):
    trim_size = round(0.1 * fs)
    mean_value = np.mean(signal[trim_size:-trim_size])
    signal[:trim_size] = mean_value
    signal[-trim_size:] = mean_value
    return signal


def find_peaks_above_threshold(signal, threshold):
    peaks, _ = find_peaks(signal, height=threshold)
    return peaks


def t_wave_detection(ecg_signal, fs):
    # Placeholder for T-wave detection, assuming peaks with larger intervals
    distance = int(0.8 * fs)  # Minimum distance between T-waves (800ms)
    t_wave_peaks, _ = find_peaks(ecg_signal, distance=distance)
    return t_wave_peaks


def p_wave_detection(ecg_signal, fs):
    # Placeholder for P-wave detection, assuming peaks with smaller intervals
    distance = int(0.2 * fs)  # Minimum distance between P-waves (200ms)
    p_wave_peaks, _ = find_peaks(ecg_signal, distance=distance)
    return p_wave_peaks
