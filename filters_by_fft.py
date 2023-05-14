from scipy.fftpack import fft, fftfreq, ifft, fftshift, ifftshift

def low_pass_filter(cutoff_freq , signal , sampling_rate):
    spectrum = fftshift(fft(signal))
    freq = fftshift(fftfreq(signal.shape[-1], 1 / sampling_rate))
    spectrum[0:(round((len(spectrum) * (sampling_rate / 2 - cutoff_freq)) / sampling_rate) + 1)] = 0
    spectrum[(signal.shape[-1] - round((len(spectrum) * (sampling_rate / 2 - cutoff_freq)) / sampling_rate)): (signal.shape[-1])] = 0
    filter_signal = ifft(ifftshift(spectrum))
    return filter_signal, freq, spectrum

def high_pass_filter(cutoff_freq , signal , sampling_rate):
    freq = fftshift(fftfreq(signal.shape[-1], 1 / sampling_rate))
    spectrum_without_shift = (fft(signal))
    spectrum_without_shift[0:(round((len(spectrum_without_shift) * cutoff_freq) / sampling_rate) + 1)] = 0
    spectrum_without_shift[(signal.shape[-1] - round((len(spectrum_without_shift) * cutoff_freq) / sampling_rate)): (signal.shape[-1])] = 0
    filter_signal = ifft(spectrum_without_shift)
    return filter_signal, freq, fftshift(spectrum_without_shift)


def band_pass_filter(cutoff_freq_down, cutoff_freq_up, signal, sampling_rate):
    signal_after_low_pass, freq, spectrum = low_pass_filter(cutoff_freq_up, signal, sampling_rate)
    filter_signal, freq, spectrum = high_pass_filter(cutoff_freq_down, signal, sampling_rate)
    return filter_signal, freq, spectrum
