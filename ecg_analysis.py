import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from lead_extractor import extract_leads


# Improved function names and added docstrings for clarity

def load_data(file_path):
    """
    Load ECG data from a given file.
    """
    return extract_leads(file_path)


def process_ecg_data(data):
    """
    Process ECG data to extract relevant leads and perform necessary pre-processing.
    """
    lead_I = data['I']
    lead_II = data['II']
    return lead_I, lead_II


def detect_peaks(ecg_signal, distance=150, height=None):
    """
    Detect peaks in the ECG signal using the find_peaks method from scipy.
    """
    peaks, _ = find_peaks(ecg_signal, distance=distance, height=height)
    return peaks


def calculate_heart_rate(peaks, fs):
    """
    Calculate the heart rate from detected peaks.
    """
    rr_intervals = np.diff(peaks) / fs * 1000  # Convert to milliseconds
    heart_rate = 60000 / np.mean(rr_intervals)
    return heart_rate


def plot_ecg(lead_I, lead_II, peaks_I, peaks_II):
    """
    Plot ECG leads with detected peaks.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(211)
    plt.plot(lead_I, label='Lead I')
    plt.plot(peaks_I, lead_I[peaks_I], 'rx')
    plt.title('Lead I with Detected Peaks')
    plt.legend()

    plt.subplot(212)
    plt.plot(lead_II, label='Lead II')
    plt.plot(peaks_II, lead_II[peaks_II], 'rx')
    plt.title('Lead II with Detected Peaks')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main(file_path, fs=500):
    """
    Main function to execute ECG analysis workflow.
    """
    # Load and process data
    data = load_data(file_path)
    lead_I, lead_II = process_ecg_data(data)

    # Detect peaks
    peaks_I = detect_peaks(lead_I)
    peaks_II = detect_peaks(lead_II)

    # Calculate heart rate
    hr_I = calculate_heart_rate(peaks_I, fs)
    hr_II = calculate_heart_rate(peaks_II, fs)

    print(f"Heart Rate from Lead I: {hr_I:.2f} bpm")
    print(f"Heart Rate from Lead II: {hr_II:.2f} bpm")

    # Plot ECG signals with detected peaks
    plot_ecg(lead_I, lead_II, peaks_I, peaks_II)


if __name__ == '__main__':
    file_path = 'path_to_ecg_file'  # Update this path
    main(file_path)
