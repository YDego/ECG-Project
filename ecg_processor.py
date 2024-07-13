import numpy as np
import processing_functions as pf
import matplotlib.pyplot as plt


class ECGProcessor:
    def __init__(self, ecg_record, signal_len):
        self.ecg_record = ecg_record
        self.signal_len = signal_len
        self.segmented_signals = []
        self.fft_segmented = []
        self.frequency_bins_segmented = []
        self.annotation_sample_segmented = []
        self.ann_markers_segmented = []
        self.num_of_segments = 0

    def process(self):
        ecg_signal = pf.baseline_removal_moving_median(self.ecg_record.ecg_signal, self.ecg_record.fs)
        is_not_full = True

        while is_not_full:
            start_cut = self.num_of_segments * self.signal_len * self.ecg_record.fs
            end_cut = (self.num_of_segments + 1) * self.signal_len * self.ecg_record.fs - 1

            if start_cut >= len(ecg_signal):
                break
            elif end_cut > len(ecg_signal):
                end_cut = len(ecg_signal)
                is_not_full = False

            ecg_signal_cut = ecg_signal[start_cut:end_cut]
            ann_cut_start = np.argmin(np.abs(np.array(self.ecg_record.annotations.sample) - start_cut))
            ann_cut_end = np.argmin(np.abs(np.array(self.ecg_record.annotations.sample) - end_cut))
            annotation_sample_cut = self.ecg_record.annotations.sample[ann_cut_start:ann_cut_end]
            ann_markers_cut = self.ecg_record.annotations.symbol[ann_cut_start:ann_cut_end]

            fft, frequency_bins = pf.compute_fft(ecg_signal_cut, self.ecg_record.fs)

            self.segmented_signals.append(ecg_signal_cut)
            self.fft_segmented.append(fft)
            self.frequency_bins_segmented.append(frequency_bins)
            self.annotation_sample_segmented.append(annotation_sample_cut)
            self.ann_markers_segmented.append(ann_markers_cut)

            self.num_of_segments += 1

        return self._create_ecg_dict()

    def _create_ecg_dict(self):
        return {
            "dataset": self.ecg_record.dataset.name,
            "record": self.ecg_record.record,
            "original_signal": self.segmented_signals,
            "signal": self.segmented_signals,
            "name": self.ecg_record.record_name,
            "ann": self.annotation_sample_segmented,
            "ann_markers": self.ann_markers_segmented,
            "our_ann": [],
            "our_ann_markers": [],
            "r_peak_success": [0, 0],
            "lead": self.ecg_record.dataset.leads[0],
            "fs": self.ecg_record.fs,
            "fft": self.fft_segmented,
            "frequency_bins": self.frequency_bins_segmented,
            "num_of_segments": self.num_of_segments,
            "signal_len": self.signal_len
        }

    def plot_first_segment(self):
        if not self.segmented_signals:
            print("No segments available to plot.")
            return

        plt.figure(figsize=(12, 10))

        # Plot original signal
        plt.subplot(2, 1, 1)
        plt.plot(self.ecg_record.ecg_signal[:self.signal_len * self.ecg_record.fs], label='Original ECG Signal')
        for sample, marker in zip(self.ecg_record.annotations.sample, self.ecg_record.annotations.symbol):
            if sample < self.signal_len * self.ecg_record.fs:
                plt.annotate(marker, (sample, self.ecg_record.ecg_signal[sample]), color='blue')
        plt.title(f"Original ECG Signal Segment - {self.ecg_record.record_name}")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.legend()

        # Plot processed signal
        plt.subplot(2, 1, 2)
        plt.plot(self.segmented_signals[0], label='Processed ECG Signal')
        for sample, marker in zip(self.annotation_sample_segmented[0], self.ann_markers_segmented[0]):
            plt.annotate(marker, (sample - self.num_of_segments * self.signal_len * self.ecg_record.fs, self.segmented_signals[0][sample - self.num_of_segments * self.signal_len * self.ecg_record.fs]), color='red')
        plt.title(f"Processed ECG Signal Segment - {self.ecg_record.record_name}")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.legend()

        # Adding the legend for annotations
        plt.subplot(2, 1, 1)
        plt.scatter([], [], color='blue', label='Original Annotations')
        plt.scatter([], [], color='red', label='Processed Annotations')
        plt.legend(loc='upper right')

        plt.subplot(2, 1, 2)
        plt.scatter([], [], color='blue', label='Original Annotations')
        plt.scatter([], [], color='red', label='Processed Annotations')
        plt.legend(loc='upper right')

        plt.tight_layout()
        plt.show()
