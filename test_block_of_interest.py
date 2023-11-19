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

success = 0
number_of_t_dots = 0
count = 0
w1_size = 0.070
dataset = 'ludb'
data = []
time_per_record = []
signal_len_in_time = 900
t_peak_location = []

for i in range(1, 200, 1):

    #if i in [38, 71, 88, 95, 101, 109, 111] and dataset == 'ludb': #
    #   continue
    start = time.time()
    erase_last_real_dot = False
    if dataset == 'qt':
        erase_last_real_dot = True

    ecg_dict_original = le.ecg_lead_ext(signal_len_in_time, dataset, i, 'ii')
    #ecg_dict_manually_ann = le.ecg_lead_ext(signal_len_in_time, dataset, i, 'ii', 'q1c')
    fs = ecg_dict_original['fs']
    ecg_dict_copy = copy.deepcopy(ecg_dict_original)
    for seg in range(ecg_dict_copy['num_of_segments']):
        #t_peaks_manually = t_wave_detection.t_peaks_annotations(ecg_dict_manually_ann, 'real', seg)
        #if t_peaks_manually.size == 0:
        #    continue
        signal = ecg_dict_copy['original_signal'][seg]
        #print(ecg_dict_original['ann_markers'][seg])
        #signal = pf.band_pass_filter(0.5, 10, signal, fs)
        b, a = scipy.signal.butter(2, [0.5, 10] ,btype='bandpass', output='ba', fs = fs)
        signal = scipy.signal.filtfilt(b, a, signal)

        q_ann, s_ann = qrs.find_q_s_ann(ecg_dict_original, seg, True, True, realLabels=True)
        if q_ann.size <= 1 or s_ann.size <= 1:
            #print(f'there is {q_ann.size} q annotations and {s_ann.size} s annotations in record {i}')
            continue

        signal_without_qrs = t_wave_detection.qrs_removal(signal, seg, q_ann, s_ann)
        #pm.plot_signal_with_dots2(signal, s_ann, q_ann, fs, 'signal', 's ann', 'q ann', i)

        r_peaks = qrs.r_peaks_annotations(ecg_dict_original, 'real', seg)
        if r_peaks.size <= 1:
            print(f'there is only {r_peaks.size} peaks')
            continue
        k_factor = 1
        #all_dots = np.array(ecg_dict_manually_ann['ann'][seg])

        # ecg_signal_filtered = pf.band_pass_filter(0.5, 25, ecg_dict_original['original_signal'][seg].copy(), fs)
        b, a = scipy.signal.butter(2, [0.5, 25], btype='bandpass', output='ba', fs=fs)
        ecg_signal_filtered = scipy.signal.filtfilt(b, a, ecg_dict_original['original_signal'][seg])
        #pm.plot_signal_with_dots(ecg_signal_filtered, all_dots, fs)
        #b, a = scipy.signal.butter(2, [0.5, 50], btype='bandpass', output='ba', fs=fs)
        #signal_without_dc = scipy.signal.filtfilt(b, a, ecg_dict_original['original_signal'][seg])
        signal_without_dc = ecg_dict_original['original_signal'][seg]
        #signal_without_dc = pf.baseline_removal_moving_median(ecg_dict_original['original_signal'][seg].copy(), fs)
        t_start, t_peak_normal, t_end, quality_factors = t_wave_detection.t_peak_detection(signal_without_qrs, fs, w1_size, k_factor,
                                                                          r_peaks, ecg_signal_filtered, 0.7, 0.15)
        t_start_low, t_peak_low, t_end_low, quality_factors_low = t_wave_detection.t_peak_detection(-signal_without_qrs, fs, w1_size,
                                                                               k_factor, r_peaks, -ecg_signal_filtered,
                                                                               0.7, 0.15)
        print(np.average(quality_factors))
        print(np.average(quality_factors_low))

        # ratio_factor = 1.5
        # t_peak = np.zeros(t_peak_normal.size, dtype=int)
        # if t_peak_low.size == t_peak_normal.size:
        #     #pm.plot_signal_with_dots2(signal_without_qrs, t_peak_low, t_peak_normal, fs, 'original signal', 'low t peaks', 'our t peaks', i, seg, signal_len_in_time)
        #     waiting_queue = []
        #     normal_peak = 0
        #     inverted_peak = 0
        #     signal_moving_average = t_wave_detection.moving_average(signal_without_dc, 0.2 * fs + 1)
        #     #pm.plot_2_signals(signal_without_dc, signal_moving_average, fs, label1='signal_without_dc', label2='signal_moving_average')
        #     for index in range(t_peak.size):
        #         if t_peak_low[index] == -1:
        #             t_peak[index] = t_peak_normal[index]
        #             normal_peak += 1
        #             continue
        #         elif t_peak_normal[index] == -1:
        #             t_peak[index] = t_peak_low[index]
        #             inverted_peak += 1
        #             continue
        #         elif t_peak_normal[index] == -1 and t_peak_low[index] == -1:
        #             t_peak[index] = 0
        #             continue
        #         rr_interval = r_peaks[index + 1] - r_peaks[index]
        #         #rt_min = int(np.ceil(0.17 * rr_interval))
        #         rt_max = int(np.ceil(0.5 * rr_interval))
        #         #x = np.abs(signal_without_dc[t_peak_normal[index]] - signal_moving_average[t_peak_normal[index]])
        #         #y = np.abs(signal_without_dc[t_peak_low[index]] - signal_moving_average[t_peak_low[index]])
        #         #x_norm_sum = np.sum(np.abs(signal_without_dc[t_start[index]:t_end[index]] - signal_moving_average[t_start[index]:t_end[index]])) / (t_end[index] - t_start[index])
        #         #y_norm_sum = np.sum(np.abs(signal_without_dc[t_start_low[index]:t_end_low[index]] - signal_moving_average[t_start_low[index]:t_end_low[index]]) / (t_end_low[index] - t_start_low[index]))
        #         if ratio_factor * np.abs(signal_without_dc[t_peak_normal[index]] - signal_moving_average[t_peak_normal[index]]) < np.abs(signal_without_dc[t_peak_low[index]] - signal_moving_average[t_peak_low[index]]):
        #             t_peak[index] = t_peak_low[index]
        #             inverted_peak += 1
        #         elif np.abs(signal_without_dc[t_peak_normal[index]] - signal_moving_average[t_peak_normal[index]]) > np.abs(signal_without_dc[t_peak_low[index]] - signal_moving_average[t_peak_low[index]]) * ratio_factor:
        #             t_peak[index] = t_peak_normal[index]
        #             normal_peak += 1
        #         else:  # check it TODO
        #             waiting_queue.append(index)
        #             #print('not decide')
        #
        #     for index in waiting_queue:
        #         if normal_peak >= inverted_peak and normal_peak != 0:
        #             t_peak[index] = t_peak_normal[index]
        #         elif inverted_peak > normal_peak or signal_without_dc[t_peak_low[index]] < 0:
        #             t_peak[index] = t_peak_low[index]
        #         else:
        #             t_peak[index] = t_peak_normal[index]

        t_real_peaks = t_wave_detection.t_peaks_annotations(ecg_dict_original, 'real', seg)

        if t_real_peaks.size == 0:
            print(f'there is 0 t peaks')
            continue
        if erase_last_real_dot:
            t_real_peaks = np.delete(t_real_peaks, t_real_peaks.size - 1)

        t_united = np.sort(np.concatenate((t_peak_normal, t_peak_low)))
        success_record, number_of_t_dots_record = t_wave_detection.comparison_t_peaks(t_real_peaks.copy(), t_united.copy(), fs, r_peaks.size - 1, 0.050)
        success += success_record
        # number_of_t_dots_record = r_peaks.size - 1
        number_of_t_dots += number_of_t_dots_record
        if number_of_t_dots_record == 0:
            continue
        score = np.round(100 * success_record / number_of_t_dots_record, 5)

        sub_data = [ecg_dict_copy['name'], f'{i}', f'{number_of_t_dots_record}', f'{success_record}', f'{score}']
        data.append(sub_data)
        # print(i, f'{success_record}/{number_of_t_dots_record}')
        if score < 95:
            count += 1
        print(i, f'{success_record}/{number_of_t_dots_record}')
        end = time.time()
        time_per_record.append(end - start)
            # pm.plot_signal_with_dots2(signal_without_dc, t_peak_normal, t_peak_low, fs, 'original signal', 't noraml peaks',
            #                            't low peaks', i, seg, signal_len_in_time)
        # pm.plot_signal_with_dots2(signal_without_dc, t_real_peaks, t_united, fs, 'original signal', 't_real_peaks', 'our t peaks', i, seg, signal_len_in_time)
if number_of_t_dots != 0:
    print(round(success * 100 / number_of_t_dots, 5), f'{success}/{number_of_t_dots}', count)

# header = ['record name', 'record number', 'number of beats', 'True Positive', 'Score']
#
# with open('qt score manually annotations 30 ms.csv', 'w', encoding='UTF8', newline='') as f:
#     writer = csv.writer(f)
#
#     # write the header
#     writer.writerow(header)
#
#     # write multiple rows
#     writer.writerows(data)

total_time = sum(time_per_record)
print(f'{total_time} sec')
#print(f'average time {total_time / i}')
print(time_per_record)

