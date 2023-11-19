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
t_peak_location = []
for i in range(1, 201, 1):
    signal_len_in_time = 10
    ecg_dict_original = le.ecg_lead_ext(signal_len_in_time, 'ludb', i, 'ii')
    fs = ecg_dict_original['fs']
    ecg_dict_copy = copy.deepcopy(ecg_dict_original)
    for seg in range(ecg_dict_copy['num_of_segments']):
        signal = ecg_dict_copy['original_signal'][seg]
        # signal = pf.band_pass_filter(0.5, 12, signal, fs)
        b, a = scipy.signal.butter(2, [0.5, 12], btype='bandpass', output='ba', fs=fs)
        signal = scipy.signal.filtfilt(b, a, signal)
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

        # ecg_signal_filtered = pf.band_pass_filter(0.5, 25, ecg_dict_original['original_signal'][seg].copy(), fs)
        b, a = scipy.signal.butter(2, [0.5, 25], btype='bandpass', output='ba', fs=fs)
        ecg_signal_filtered = scipy.signal.filtfilt(b, a, ecg_dict_original['original_signal'][seg])
        # b, a = scipy.signal.butter(2, [0.5, fs/2 -1], btype='bandpass', output='ba', fs=fs)
        # signal_without_dc = scipy.signal.filtfilt(b, a, ecg_dict_original['original_signal'][seg])
        signal_without_dc = ecg_dict_original['original_signal'][seg]
        # signal_without_dc = pf.baseline_removal_moving_median(ecg_dict_original['original_signal'][seg].copy(), fs)
        t_start, t_peak_maxima, t_end = t_wave_detection.t_peak_detection(signal_without_qrs, fs, w1_size, k_factor,
                                                                          r_peaks, ecg_signal_filtered)
        t_start_minima, t_peak_minima, t_end_minima = t_wave_detection.t_peak_detection(-signal_without_qrs, fs,
                                                                                        w1_size,
                                                                                        2 * k_factor, r_peaks,
                                                                                        -ecg_signal_filtered,
                                                                                        0.6, 0.17)

        our_t_peak = np.zeros(t_peak_maxima.size, dtype=int)

        if t_peak_minima.size == t_peak_maxima.size:
            # pm.plot_signal_with_dots2(signal_without_qrs, t_peak_minima, t_peak_maxima, fs, 'original signal', 'minima t peaks',
            # 'our t peaks', i, seg, signal_len_in_time)
            maxima_peak = 0
            minima_peak = 0
            decision_threshold = 0.3
            maxima_factor = 1.5
            minima_factor = 1
            signal_moving_average = t_wave_detection.moving_average(signal_without_dc, fs * 0.14 + 1)
            repeating_list = np.arange(0, our_t_peak.size, 1)
            removing_list = []
            # pm.plot_2_signals(signal_without_dc, signal_moving_average, fs, label1='signal_without_dc',
            #                 label2='signal_moving_average')
            while not len(repeating_list) == 0:
                for index in repeating_list:
                    mean = 0
                    if t_peak_maxima[index] == -1 and t_peak_minima[index] == -1:
                        our_t_peak[index] = 0
                        removing_list.append(index)
                        continue
                    elif t_peak_minima[index] == -1:
                        our_t_peak[index] = t_peak_maxima[index]
                        maxima_peak += 1
                        removing_list.append(index)
                        continue
                    elif t_peak_maxima[index] == -1:
                        our_t_peak[index] = t_peak_minima[index]
                        minima_peak += 1
                        removing_list.append(index)
                        continue

                    rr_interval = r_peaks[index + 1] - r_peaks[index]
                    # rt_min = int(np.ceil(0.17 * rr_interval))

                    rt_max = int(np.ceil(0.5 * rr_interval))
                    # mean = np.mean(signal_without_dc[s_ann[index]:r_peaks[index] + rt_max])

                    # t_real_peaks_loc = t_wave_detection.t_peaks_annotations(ecg_dict_original, 'real', seg)
                    # for t_index in range(t_real_peaks_loc.size):
                    #     if r_peaks[index] < t_real_peaks_loc[t_index] < r_peaks[index + 1]:
                    #         t_peak_location.append(100 * (t_real_peaks_loc[t_index] - r_peaks[index]) / rr_interval)

                    # find t normalized index between r peaks
                    t_norm_index_maxima = 1
                    for t_index in range(t_peak_maxima.size):
                        if r_peaks[index] < t_peak_maxima[t_index] < r_peaks[index + 1]:
                            t_norm_index_maxima = (t_peak_maxima[t_index] - r_peaks[index]) / rr_interval

                    t_norm_index_minima = 1
                    for t_index in range(t_peak_minima.size):
                        if r_peaks[index] < t_peak_minima[t_index] < r_peaks[index + 1]:
                            t_norm_index_minima = (t_peak_minima[t_index] - r_peaks[index]) / rr_interval

                    maxima_peak_score = t_wave_detection.score_value(signal_without_dc, signal_moving_average,
                                                                     t_peak_maxima[index], t_norm_index_maxima,
                                                                     ratio_factor=maxima_factor)
                    minima_peak_score = t_wave_detection.score_value(signal_without_dc, signal_moving_average,
                                                                     t_peak_minima[index], t_norm_index_minima,
                                                                     ratio_factor=minima_factor)

                    decision = np.abs(maxima_peak_score - minima_peak_score) / max(maxima_peak_score,
                                                                                   minima_peak_score)
                    if decision < decision_threshold:
                        # t_real_peaks_loc = t_wave_detection.t_peaks_annotations(ecg_dict_original, 'real', seg)
                        # for t_index in range(t_real_peaks_loc.size):
                        #     if r_peaks[index] < t_real_peaks_loc[t_index] < r_peaks[index + 1]:
                        #         minima = np.abs(t_real_peaks_loc[t_index] - t_peak_minima[index]) < np.abs(
                        #             t_real_peaks_loc[t_index] - t_peak_maxima[index])
                        #         pm.plot_score([maxima_peak_score, minima_peak_score], minima)
                        continue
                    elif maxima_peak_score < minima_peak_score:
                        our_t_peak[index] = t_peak_minima[index]
                        removing_list.append(index)
                        minima_peak += 1
                    else:
                        our_t_peak[index] = t_peak_maxima[index]
                        removing_list.append(index)
                        maxima_peak += 1
                if maxima_peak == 0 and minima_peak == 0:
                    decision_threshold = decision_threshold * 1.2
                    continue
                if maxima_peak > minima_peak:
                    maxima_factor = maxima_factor * (1 + maxima_peak / (maxima_peak + minima_peak))
                else:
                    minima_factor = minima_factor * (1 + minima_peak / (maxima_peak + minima_peak))
                removing_list_index = []
                for value in removing_list:
                    removing_list_index.append(np.argwhere(repeating_list == value))
                repeating_list = np.delete(repeating_list, removing_list_index)
                removing_list_index = []
                removing_list = []

        t_real_peaks = t_wave_detection.t_peaks_annotations(ecg_dict_original, 'real', seg)
        success_record, number_of_t_dots_record = t_wave_detection.comparison_t_peaks(t_real_peaks.copy(),
                                                                                      our_t_peak.copy(), fs)
        success += success_record
        number_of_t_dots += our_t_peak.size
        # print(i, f'{success_record}/{number_of_t_dots_record}')
        if np.round(100 * success_record / our_t_peak.size) < 100:
            count += 1
            print(i, f'{success_record}/{our_t_peak.size}')

        # pm.plot_signal_with_dots2(signal_without_dc, t_real_peaks, our_t_peak, fs, 'original signal',
        #                               't_real_peaks', 'our t peaks', i, seg, signal_len_in_time)

        # pm.plot_signal_with_dots3(signal_without_dc, t_real_peaks, t_peak_maxima, t_peak_minima, fs, 'original signal',
        #                           't_real_peaks', 'our t peaks maxima', 'our t peaks minima', i, seg, signal_len_in_time)
print(round(success * 100 / number_of_t_dots, 5), f'{success}/{number_of_t_dots}', count)
# pm.plot_hist(t_peak_location)
