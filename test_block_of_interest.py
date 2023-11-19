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
        t_start, t_peak_normal, t_end = t_wave_detection.t_peak_detection(signal_without_qrs, fs, w1_size, k_factor,
                                                                          r_peaks, ecg_signal_filtered)
        t_start_low, t_peak_low, t_end_low = t_wave_detection.t_peak_detection(-signal_without_qrs, fs, w1_size,
                                                                               2 * k_factor, r_peaks,
                                                                               -ecg_signal_filtered,
                                                                               0.6, 0.17)

        our_t_peak = np.zeros(t_peak_normal.size, dtype=int)
        if t_peak_low.size == t_peak_normal.size:
            # pm.plot_signal_with_dots2(signal_without_qrs, t_peak_low, t_peak_normal, fs, 'original signal', 'low t peaks',
            # 'our t peaks', i, seg, signal_len_in_time)
            waiting_queue = []
            normal_peak = 0
            inverted_peak = 0
            signal_moving_average = t_wave_detection.moving_average(signal_without_dc, fs * 0.14 + 1)
            # pm.plot_2_signals(signal_without_dc, signal_moving_average, fs, label1='signal_without_dc',
            #                 label2='signal_moving_average')
            for index in range(our_t_peak.size):
                mean = 0
                if t_peak_low[index] == -1:
                    our_t_peak[index] = t_peak_normal[index]
                    normal_peak += 1
                    continue
                elif t_peak_normal[index] == -1:
                    our_t_peak[index] = t_peak_low[index]
                    inverted_peak += 1
                    continue
                elif t_peak_normal[index] == -1 and t_peak_low[index] == -1:
                    our_t_peak[index] = 0
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
                t_norm_index_high = 1
                for t_index in range(t_peak_normal.size):
                    if r_peaks[index] < t_peak_normal[t_index] < r_peaks[index + 1]:
                        t_norm_index_high = (t_peak_normal[t_index] - r_peaks[index]) / rr_interval

                t_norm_index_low = 1
                for t_index in range(t_peak_low.size):
                    if r_peaks[index] < t_peak_low[t_index] < r_peaks[index + 1]:
                        t_norm_index_low = (t_peak_low[t_index] - r_peaks[index]) / rr_interval

                normal_peak_score = t_wave_detection.score_value(signal_without_dc, signal_moving_average,
                                                                 t_peak_normal[index], t_norm_index_high, ratio_factor=1.5)
                inverted_peak_score = t_wave_detection.score_value(signal_without_dc, signal_moving_average,
                                                                   t_peak_low[index], t_norm_index_low, ratio_factor=1)

                t_real_peaks_loc = t_wave_detection.t_peaks_annotations(ecg_dict_original, 'real', seg)
                for t_index in range(t_real_peaks_loc.size):
                    if r_peaks[index] < t_real_peaks_loc[t_index] < r_peaks[index + 1]:
                        inverted = np.abs(t_real_peaks_loc[t_index]-t_peak_low[index]) < np.abs(t_real_peaks_loc[t_index]-t_peak_normal[index])
                        pm.plot_score([normal_peak_score, inverted_peak_score], inverted)

                if normal_peak_score < inverted_peak_score:
                    our_t_peak[index] = t_peak_low[index]
                    inverted_peak += 1
                else:
                    our_t_peak[index] = t_peak_normal[index]
                    normal_peak += 1

        t_real_peaks = t_wave_detection.t_peaks_annotations(ecg_dict_original, 'real', seg)
        success_record, number_of_t_dots_record = t_wave_detection.comparison_t_peaks(t_real_peaks.copy(),
                                                                                      our_t_peak.copy(), fs)
        success += success_record
        number_of_t_dots += our_t_peak.size
        # print(i, f'{success_record}/{number_of_t_dots_record}')
        if np.round(100 * success_record / our_t_peak.size) < 100:
            count += 1
            print(i, f'{success_record}/{our_t_peak.size}')

        pm.plot_signal_with_dots2(signal_without_dc, t_real_peaks, our_t_peak, fs, 'original signal',
                                  't_real_peaks', 'our t peaks', i, seg, signal_len_in_time)

        # pm.plot_signal_with_dots3(signal_without_dc, t_real_peaks, t_peak_normal, t_peak_low, fs, 'original signal',
        #                           't_real_peaks', 'our t peaks high', 'our t peaks low', i, seg, signal_len_in_time)
print(round(success * 100 / number_of_t_dots, 5), f'{success}/{number_of_t_dots}', count)
# pm.plot_hist(t_peak_location)
