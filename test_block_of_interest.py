import numpy as np
import lead_extractor as le
import plot_manager as pm
import qrs_detection as qrs
import copy
import csv
import t_wave_detection
import processing_functions as pf
import scipy


# check signals 121 124 .


success = 0
number_of_t_dots = 0
count = 0
w1_size = 0.070
for i in range(1, 201, 1):
    signal_len_in_time = 10
    ecg_dict_original = le.ecg_lead_ext(signal_len_in_time, 'ludb', i, 'ii')
    fs = ecg_dict_original['fs']
    ecg_dict_copy = copy.deepcopy(ecg_dict_original)
    for seg in range(ecg_dict_copy['num_of_segments']):
        signal = ecg_dict_copy['original_signal'][seg]
        #signal = pf.band_pass_filter(0.5, 12, signal, fs)
        b, a = scipy.signal.butter(2, [0.5, 12] ,btype='bandpass', output='ba', fs = fs)
        signal = scipy.signal.filtfilt(b, a, signal)  # could use also "lfilter" function
        q_ann, s_ann = qrs.find_q_s_ann(ecg_dict_original, seg, True, True, realLabels=True)
        if q_ann.size <= 1 or s_ann.size <= 1:
            print(f'there is {q_ann.size} q annotations and {s_ann.size} s annotations in record {i}')
            continue
        signal_without_qrs = t_wave_detection.qrs_removal(signal, seg, q_ann, s_ann)
        r_peaks = qrs.r_peaks_annotations(ecg_dict_original, 'real', seg)
        if r_peaks.size <= 1:
            print(f'there is only {r_peaks.size} peaks')
            continue
        k_factor = 1

        #ecg_signal_filtered = pf.band_pass_filter(0.5, 25, ecg_dict_original['original_signal'][seg].copy(), fs)
        b, a = scipy.signal.butter(2, [0.5, 25], btype='bandpass', output='ba', fs=fs)
        ecg_signal_filtered = scipy.signal.filtfilt(b, a, ecg_dict_original['original_signal'][seg])
        #b, a = scipy.signal.butter(2, [0.5, fs/2 -1], btype='bandpass', output='ba', fs=fs)
        #signal_without_dc = scipy.signal.filtfilt(b, a, ecg_dict_original['original_signal'][seg])
        signal_without_dc = ecg_dict_original['original_signal'][seg]
        #signal_without_dc = pf.baseline_removal_moving_median(ecg_dict_original['original_signal'][seg].copy(), fs)
        t_start, t_peak, t_end = t_wave_detection.t_peak_detection(signal_without_qrs, fs, w1_size, k_factor, r_peaks, ecg_signal_filtered, signal)
        t_start_low, t_peak_low, t_end_low = t_wave_detection.t_peak_detection(-signal_without_qrs, fs, w1_size,
                                                                               k_factor, r_peaks, -ecg_signal_filtered,
                                                                               -signal, 0.4, 0.25)

        ratio_factor = 3
        if t_peak_low.size == t_peak.size:
            for index in range(t_peak.size):
                if ratio_factor * signal_without_dc[t_peak[index]] < np.abs(signal_without_dc[t_peak_low[index]]):
                    t_peak[index] = t_peak_low[index]
                    ratio_factor /= 1.1
                else:

                    ratio_factor *= 1.2
        t_real_peaks = t_wave_detection.t_peaks_annotations(ecg_dict_original, 'real', seg)
        success_record, number_of_t_dots_record = t_wave_detection.comparison_t_peaks(t_real_peaks.copy(), t_peak.copy(), fs)
        success += success_record
        number_of_t_dots += t_peak.size
        # print(i, f'{success_record}/{number_of_t_dots_record}')
        if np.round(100 * success_record / number_of_t_dots_record) < 100:
            count += 1
            print(i, f'{success_record}/{number_of_t_dots_record}')
        #pm.plot_signal_with_dots2(signal_without_dc, t_real_peaks, t_peak, fs, 'original signal', 'real t peaks', 'our t peaks', i, seg, signal_len_in_time)
print(round(success * 100 / number_of_t_dots, 5), f'{success}/{number_of_t_dots}', count)

