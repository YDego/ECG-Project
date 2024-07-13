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
        self.qrs_indices = []
        self.num_of_segments = 0

    def process(self):
        self.pre_process_signal()
        self.detect_qrs()
        self.detect_t_wave()
        self.detect_p_wave()
        return self._create_ecg_dict()

    def pre_process_signal(self):
        ecg_signal = pf.baseline_removal_moving_median(self.ecg_record.ecg_signal, self.ecg_record.fs)
        self.segmented_signals = self.segment_signal(ecg_signal)
        self.plot_segment(self.segmented_signals[0], title="Pre-Processed ECG Signal", annotations=self.annotation_sample_segmented[0], markers=self.ann_markers_segmented[0])

    def detect_qrs(self):
        self.qrs_indices = pf.qrs_detection(self.segmented_signals[0], self.ecg_record.fs)
        self.plot_segment(self.segmented_signals[0], title="QRS Detection", annotations=self.qrs_indices, markers=['R'] * len(self.qrs_indices))

    def detect_t_wave(self):
        t_wave_indices = pf.t_wave_detection(self.segmented_signals[0], self.ecg_record.fs, self.qrs_indices)
        self.plot_segment(self.segmented_signals[0], title="T Wave Detection", annotations=t_wave_indices, markers=['T'] * len(t_wave_indices))

    def detect_p_wave(self):
        p_wave_indices = pf.p_wave_detection(self.segmented_signals[0], self.ecg_record.fs)
        self.plot_segment(self.segmented_signals[0], title="P Wave Detection", annotations=p_wave_indices, markers=['P'] * len(p_wave_indices))

    def segment_signal(self, ecg_signal):
        is_not_full = True
        segmented_signals = []

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

            self.annotation_sample_segmented.append(annotation_sample_cut)
            self.ann_markers_segmented.append(ann_markers_cut)
            segmented_signals.append(ecg_signal_cut)

            self.num_of_segments += 1

        return segmented_signals

    @staticmethod
    def plot_segment(signal, title, annotations=None, markers=None):
        plt.figure(figsize=(10, 4))
        plt.plot(signal, label='Processed ECG Signal')
        if annotations is not None and markers is not None and len(annotations) > 0 and len(markers) > 0:
            for sample, marker in zip(annotations, markers):
                plt.annotate(marker, (sample, signal[sample]), color='red')
        plt.title(title)
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.show()

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
