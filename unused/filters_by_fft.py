# import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftfreq, ifft, fftshift, ifftshift


def low_pass_filter(cutoff_freq , signal , sampling_rate):
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
    return filter_signal, freq, fftshift(spectrum_without_shift)


def notch_filter(frequencies, bandwidth, signal, sampling_rate):
    freq = np.fft.fftshift(np.fft.fftfreq(signal.shape[-1], 1 / sampling_rate))
    spectrum_without_shift = np.fft.fft(signal)

    for center_freq in frequencies:
        # Find the indices of the frequencies within the specified notch region
        notch_indices = np.where((freq >= center_freq - bandwidth / 2) & (freq <= center_freq + bandwidth / 2))

        # Set the spectrum values within the notch region to 0
        spectrum_without_shift[notch_indices] = 0

    filter_signal = np.fft.ifft(spectrum_without_shift)
    return filter_signal, freq, np.fft.fftshift(spectrum_without_shift)




