import numpy as np
import lead_extractor as le
import plot_manager as pm
import qrs_detection as qrs
import copy
import csv
import t_wave_detection
import processing_functions as pf
import scipy
import time

success_per_seg = 0
success_per_peak = 0
number_of_t_dots = 0
number_of_segments = 0
count = 0
w1_size = 0.070
dataset = 'ludb'
data = []
time_per_record = []
signal_len_in_time = 900

for i in range(1, 200, 1):

    # if i in [38, 71, 88, 95, 101, 109, 111] and dataset == 'ludb': #
    #   continue
    start = time.time()
    erase_last_real_dot = False
    if dataset == 'qt':
        erase_last_real_dot = True

    ecg_dict_original = le.ecg_lead_ext(signal_len_in_time, dataset, i, 'ii')
    # ecg_dict_manually_ann = le.ecg_lead_ext(signal_len_in_time, dataset, i, 'ii', 'q1c')
    fs = ecg_dict_original['fs']
    ecg_dict_copy = copy.deepcopy(ecg_dict_original)
    all_peaks_correct = True
    for seg in range(ecg_dict_copy['num_of_segments']):
        # t_peaks_manually = t_wave_detection.t_peaks_annotations(ecg_dict_manually_ann, 'real', seg)
        # if t_peaks_manually.size == 0:
        #    continue
        signal = ecg_dict_copy['original_signal'][seg]
        # print(ecg_dict_original['ann_markers'][seg])
        # signal = pf.band_pass_filter(0.5, 10, signal, fs)
        b, a = scipy.signal.butter(2, [0.5, 10], btype='bandpass', output='ba', fs=fs)
        signal = scipy.signal.filtfilt(b, a, signal)

        q_ann, s_ann = qrs.find_q_s_ann(ecg_dict_original, seg, True, True, realLabels=True)
        if q_ann.size <= 1 or s_ann.size <= 1:
            # print(f'there is {q_ann.size} q annotations and {s_ann.size} s annotations in record {i}')
            continue

        signal_without_qrs = t_wave_detection.qrs_removal(signal, seg, q_ann, s_ann)
        # pm.plot_signal_with_dots2(signal, s_ann, q_ann, fs, 'signal', 's ann', 'q ann', i)

        r_peaks = qrs.r_peaks_annotations(ecg_dict_original, 'real', seg)
        if r_peaks.size <= 1:
            print(f'there is only {r_peaks.size} peaks')
            continue
        k_factor = 1
        # all_dots = np.array(ecg_dict_manually_ann['ann'][seg])

        # ecg_signal_filtered = pf.band_pass_filter(0.5, 25, ecg_dict_original['original_signal'][seg].copy(), fs)
        b, a = scipy.signal.butter(2, [0.5, 25], btype='bandpass', output='ba', fs=fs)
        ecg_signal_filtered = scipy.signal.filtfilt(b, a, ecg_dict_original['original_signal'][seg])
        # pm.plot_signal_with_dots(ecg_signal_filtered, all_dots, fs)
        # b, a = scipy.signal.butter(2, [0.5, 50], btype='bandpass', output='ba', fs=fs)
        # signal_without_dc = scipy.signal.filtfilt(b, a, ecg_dict_original['original_signal'][seg])
        signal_without_dc = ecg_dict_original['original_signal'][seg]
        signal_moving_average = t_wave_detection.moving_average(signal_without_dc, 0.2 * fs + 1)
        t_real_peaks = t_wave_detection.t_peaks_annotations(ecg_dict_original, 'real', seg)

        if t_real_peaks.size == 0:
            print(f'there is 0 t peaks')
            continue
        if erase_last_real_dot:
            t_real_peaks = np.delete(t_real_peaks, t_real_peaks.size - 1)
        # signal_without_dc = pf.baseline_removal_moving_median(ecg_dict_original['original_signal'][seg].copy(), fs)

        t_start_maxima, t_peak_maxima, t_end_maxima, quality_factors_maxima = t_wave_detection.t_peak_detection(
            signal_without_qrs, fs, w1_size, k_factor,
            r_peaks, ecg_signal_filtered, 0.7, 0.15)
        t_start_minima, t_peak_minima, t_end_minima, quality_factors_minima = t_wave_detection.t_peak_detection(
            -signal_without_qrs, fs, w1_size,
            k_factor, r_peaks, -ecg_signal_filtered, 0.7, 0.15)

        qf_max_avg = np.average(quality_factors_maxima)
        qf_min_avg = np.average(quality_factors_minima)

        t_peak_location = []

        for r_idx, r_peak in enumerate(r_peaks):

            if r_idx + 1 == len(r_peaks):
                break

            next_r = r_peaks[r_idx + 1]
            t_max = t_wave_detection.find_t_between_r(r_peak, next_r, t_peak_maxima)
            t_min = t_wave_detection.find_t_between_r(r_peak, next_r, t_peak_minima)
            t_real = t_wave_detection.find_t_between_r(r_peak, next_r, t_real_peaks)

            if (t_min is None and t_max is None) or t_real is None:
                t_peak_location.append(-1)
                continue
            elif t_min is None:
                t_peak_location.append(t_max)
                continue
            elif t_max is None:
                t_peak_location.append(t_min)
                continue

            norm_idx_maxima = (t_max - r_peak) / (next_r - r_peak)
            norm_idx_minima = (t_min - r_peak) / (next_r - r_peak)
            inverted = np.abs(t_min-t_real) < np.abs(t_max-t_real)
            selected_peak, peak_is_correct = t_wave_detection.calculate_total_score(signal_without_dc, signal_moving_average, t_min, t_max,
                                                                                    qf_min_avg, qf_max_avg, norm_idx_minima, norm_idx_maxima, inverted)
            all_peaks_correct = all_peaks_correct and peak_is_correct
            t_peak_location.append(selected_peak)

            number_of_t_dots += 1
            success_per_peak += int(peak_is_correct)

        t_peak_location = np.array(t_peak_location)

        # if not all_peaks_correct:
        #     pm.plot_signal_with_dots2(signal_without_dc, t_real_peaks, t_peak_location, fs, 'original signal', 't_real_peaks', 'our t peaks', i, seg, signal_len_in_time)
        success_per_seg += int(all_peaks_correct)
        number_of_segments += 1

print("success per peak = " + str(round(100 * success_per_peak/number_of_t_dots, 3)))
print("success per segment = " + str(round(100 * success_per_seg/number_of_segments, 3)))

