import statistics

import qrs_detection as qrs
import plot_manager as pm
import math
import numpy as np
import lead_extractor as le
import copy
import processing_functions as pf
import scipy

## function to run without any tests
def main_t_peak_detection(ecg_dict_original, w1_size, signal_len_in_time, which_r_ann, real_q_s_ann=False):
    fs = ecg_dict_original['fs']
    ecg_dict_copy = copy.deepcopy(ecg_dict_original)
    t_peak_selected_all_seg_list = []
    for seg in range(ecg_dict_copy['num_of_segments']):
        signal = ecg_dict_copy['original_signal'][seg]
        b, a = scipy.signal.butter(2, [0.5, 10], btype='bandpass', output='ba', fs=fs)
        signal = scipy.signal.filtfilt(b, a, signal)
        q_ann, s_ann = qrs.find_q_s_ann(ecg_dict_original, seg, True, True, realLabels=real_q_s_ann)
        if q_ann.size <= 1 or s_ann.size <= 1:
            continue
        signal_without_qrs = qrs_removal(signal, seg, q_ann, s_ann)
        # pm.plot_signal_with_dots2(signal, s_ann, q_ann, fs, 'signal', 's ann', 'q ann', i)
        r_peaks = qrs.r_peaks_annotations(ecg_dict_original, which_r_ann, seg)
        if r_peaks.size <= 1:
            print(f'there is only {r_peaks.size} peaks')
            continue
        k_factor = 1
        b, a = scipy.signal.butter(2, [0.5, 25], btype='bandpass', output='ba', fs=fs)
        ecg_signal_filtered = scipy.signal.filtfilt(b, a, ecg_dict_original['original_signal'][seg])
        t_start, t_peak_normal, t_end, quality_factors = t_peak_detection_aux(signal_without_qrs, fs,
                                                                                               w1_size, k_factor,
                                                                                               r_peaks,
                                                                                               ecg_signal_filtered, 0.7,
                                                                                               0.15)

        t_start_low, t_peak_low, t_end_low, quality_factors_low = t_peak_detection_aux(
            -signal_without_qrs, fs, w1_size,
            k_factor, r_peaks, -ecg_signal_filtered,
            0.7, 0.15)
        #print(quality_factors, quality_factors_low)

        decision_threshold = 0.25  # 0 to 1

        t_peak_location = []
        for r_idx, r_peak in enumerate(r_peaks):

            if r_idx + 1 == len(r_peaks):
                break
            qf_max = quality_factors[r_idx]
            qf_min = quality_factors_low[r_idx]

            next_r = r_peaks[r_idx + 1]
            t_max = find_t_between_r(r_peak, next_r, t_peak_normal)
            t_min = find_t_between_r(r_peak, next_r, t_peak_low)

            if t_min is None and t_max is None:
                t_peak_location.append(-1)
                continue
            elif t_min is None:
                t_peak_location.append(t_max)
                continue
            elif t_max is None:
                t_peak_location.append(t_min)
                continue
            median = statistics.median(ecg_signal_filtered[r_peaks[r_idx]:r_peaks[r_idx+1]])
            #print(median)
            amp_normal = abs(ecg_signal_filtered[t_max] - median)
            amp_low = abs(ecg_signal_filtered[t_min] - median)
            t_peak_selected = t_peak_classifier(t_max, t_min, qf_max, qf_min, amp_normal, amp_low, decision_threshold)
            t_peak_location.append(t_peak_selected)

        t_peak_selected_all_seg_list.extend(np.array(t_peak_location) + seg * signal_len_in_time * fs)

    t_peak_selected_all_seg_np = np.array(t_peak_selected_all_seg_list)
    return t_peak_selected_all_seg_np


def t_peak_classifier(t_normal, t_low, qf_normal, qf_low, amp_normal, amp_low, threshold=0):
    # check if diff is lower than threshold
    qf_check = abs(qf_normal-qf_low) / max(qf_normal, qf_low) > threshold
    amp_check = abs(amp_normal-amp_low) / max(amp_normal, amp_low) > threshold
    if qf_check:
        if qf_normal > qf_low:
            return t_normal
        return t_low
    elif amp_check:
        if amp_normal > amp_low:
            return t_normal
        return t_low
    else:
        return min(t_normal, t_low)



def find_t_between_r(this_r, next_r, t_list):
    for t in t_list:
        if not this_r < t < next_r:
            continue
        else:
            return t


# page 9-10
def t_peak_detection_aux(signal_without_qrs, fs, w1_size, k_factor, r_peaks, ecg_signal_filtered_by25, d_max=0.800, d_min=0.150):
    first_boi, ma_peak, ma_t_wave = block_of_interest(signal_without_qrs, fs, w1_size, w1_size * 2, 1) # TODO k factor changed
    #pm.plot_2_signals(signal_without_qrs, first_boi, fs, 'signal without qrs', 'initial block of interests')
    new_boi = thresholding(fs, first_boi, r_peaks, w1_size, k_factor, d_max, d_min)
    #pm.plot_3_signals(new_boi, ma_peak, ma_t_wave, fs, 'new boi', 'ma_peak', 'ma t wave')
    #pm.plot_2_signals(signal_without_qrs, new_boi, fs, 'signal without qrs', 'middle block of interests')
    t_start, t_peak, t_end, factor_function_list, quality_factor_for_beats = find_real_blocks(signal_without_qrs, fs, ecg_signal_filtered_by25, r_peaks, new_boi, ma_peak, ma_t_wave, w1_size)

    # k_start = k_factor
    # if np.count_nonzero(t_peak == -1) < 0.1 * t_peak.size:
    #     for index in range(t_peak.size):    # think about localize on one interval RR
    #         k_factor = k_start
    #         while t_peak[index] == -1 and k_factor > 0.5:
    #             k_factor -= 0.1
    #             new_boi[r_peaks[index]:r_peaks[index+1]] = thresholding(fs, first_boi, r_peaks[index:index+2], w1_size, k_factor, d_max, d_min)[r_peaks[index]:r_peaks[index+1]]
    #             t_start[index], t_peak[index], t_end[index], _ = find_real_blocks(signal_without_qrs, fs, ecg_signal_filtered_by25, r_peaks[index:index+2], new_boi, ma_peak, ma_t_wave)
    #pm.plot_2_signals(signal_without_qrs, new_boi, fs, 'signal without qrs', 'final block of interests')

        # for plot factor function in every interval
    # factor_functions = np.zeros(signal_without_qrs.size, dtype=float)
    # for index in range(r_peaks.size - 1):
    #     factor_functions[r_peaks[index]:r_peaks[index + 1]] = factor_function_list[index]
    #pm.plot_3_signals(signal_without_qrs, new_boi, factor_functions, fs, 'signal without qrs', 'final block of interests', 'factor_functions')

    return t_start, t_peak, t_end, quality_factor_for_beats


def block_of_interest(signal, fs, w1_size=0.070, w2_size=0.140, k=1):
    w1_size = np.floor(w1_size * fs * k)
    w2_size = np.floor(w2_size * fs * k)
    if w1_size % 2 != 1:
        w1_size = w1_size - 1
    if w2_size % 2 != 1:
        w2_size = w2_size - 1
    ma_peak = moving_average(signal, w1_size)
    ma_t_wave = moving_average(signal, w2_size)
    n = len(signal)
    boi = np.zeros(n, dtype=int)
    for i in range(n):
        boi[i] = int(ma_peak[i] > ma_t_wave[i] and signal[i] != 0)

    return boi, ma_peak, ma_t_wave


def moving_average(arr, w_size):
    # Initialize an empty list to store moving averages
    n = len(arr)
    moving_averages = np.zeros(n, dtype=float)
    w_size = int(w_size)
    if w_size % 2 != 1: #TODO   if not  integer
        w_size = np.round(w_size)
    half_window = int((w_size - 1) / 2)
    for i in range(half_window, n - half_window):
        low_index = i - half_window
        high_index = i + half_window
        # Calculate the average of current window
        window_average = np.round(np.sum(arr[low_index:high_index + 1]) / w_size, 3)
        moving_averages[i] = window_average
    return moving_averages


def qrs_removal(signal, seg, q_ann, s_ann):
    # return signal without QRS complex (return only the signal without all the dict)
    new_signal = np.zeros(signal.size, dtype=float)

    # q_ann, s_ann = qrs_detection.find_q_s_ann(ecg_original_copy, True, True, realLabels=realLabels)
    if q_ann[0] < s_ann[0]:
        q_ann = np.delete(q_ann, 0)

    counter = 0
    while counter < q_ann.size and counter < s_ann.size:
        new_signal[s_ann[counter]:q_ann[counter]] = signal[s_ann[counter]:q_ann[counter]]
        counter = counter + 1

    # cant be more q dots than s dots

    if counter == q_ann.size and counter < s_ann.size:
        new_signal[s_ann[counter]:signal.size] = signal[s_ann[counter]:signal.size]
    elif counter == q_ann.size and counter == s_ann.size:
        pass  # nothing should be done

    return new_signal


def thresholding(fs, boi, r_peaks, w1_size=0.070, k=1, d_max=0.800, d_min=0.170):
    rr_intervals = np.diff(r_peaks)
    for i in range(0, r_peaks.size - 1):
        rt_min = int(np.ceil(d_min * rr_intervals[i]))
        rt_max = int(np.ceil(d_max * rr_intervals[i]))
        boi[r_peaks[i]: r_peaks[i] + rt_min] = 0
        boi[r_peaks[i] + rt_max: r_peaks[i + 1]] = 0

    new_boi = np.zeros(len(boi), dtype=int)
    thr = np.floor(w1_size * fs * k)
    count = 0
    for i, x in enumerate(boi):
        if x == 1:
            count += 1
        elif count >= int(thr):
            new_boi[i - count:i] = 1
            count = 0
        else:
            count = 0
    return new_boi


def find_real_blocks(original_signal, fs, signal_filtered, r_peaks, boi, ma_peak, ma_t_wave, w_size):
    t_potential_start_potential_end = np.diff(boi)  # 4999 size
    t_potential_start_potential_end = union_block_of_interest(t_potential_start_potential_end, min(int(w_size*fs), 30), max(int(w_size*fs*0.5), 10))
    t_potential_start = np.zeros(len(t_potential_start_potential_end), dtype=int)
    t_potential_end = np.zeros(len(t_potential_start_potential_end), dtype=int)
    for i, x in enumerate(t_potential_start_potential_end):
        if x == 1:
            t_potential_start[i] = i
        elif x == -1:
            t_potential_end[i] = i
        else:
            pass
    t_potential_start = t_potential_start[t_potential_start != 0]
    t_potential_end = t_potential_end[t_potential_end != 0]
    if t_potential_start.size != t_potential_end.size:
        print("error in function find_potential_t_peaks ::: t_potential_start.size != t_potential_end.size ")
        exit(1)
    else:
        # r_peaks = qrs_detection.r_peaks_annotations(ecg_copy, 'real')  # added as out function parameter
        number_of_intervals = r_peaks.size - 1
        t_start = np.zeros(number_of_intervals, dtype=int)
        t_end = np.zeros(number_of_intervals, dtype=int)
        t_peak = np.zeros(number_of_intervals, dtype=int)
        quality_factor_for_beats = np.zeros(number_of_intervals, dtype=float)
        factor_functions_list = []
        for index in range(0, number_of_intervals):
            #if index > 843:
            #    break
            #print(index, number_of_intervals)
            window_len_interval = r_peaks[index+1] - r_peaks[index]
            if window_len_interval == 0:
                continue
            t_potential_start_one_interval = t_potential_start[r_peaks[index] < t_potential_start]
            t_potential_start_one_interval = t_potential_start_one_interval[t_potential_start_one_interval < r_peaks[index + 1]]
            t_potential_end_one_interval = t_potential_end[r_peaks[index] < t_potential_end]
            t_potential_end_one_interval = t_potential_end_one_interval[t_potential_end_one_interval < r_peaks[index + 1]]
            t_start[index], t_peak[index], t_end[index], factor_function, quality_factor_for_beats[index] = detected_real_block_of_interest(original_signal, fs,
                                                                                          signal_filtered,
                                                                                          t_potential_start_one_interval,
                                                                                          t_potential_end_one_interval,
                                                                                          ma_peak,
                                                                                          ma_t_wave,
                                                                                          window_len_interval,
                                                                                          r_peaks[index])
            factor_functions_list.append(factor_function)

        return t_start[t_start != 0], t_peak[t_peak != 0], t_end[t_end != 0], factor_functions_list, quality_factor_for_beats

#                               find t_peak over ****one interval**** , return also start and end of the block interest


#                                             inner function
def detected_real_block_of_interest(original_signal, fs, signal_filtered, t_potential_start, t_potential_end, ma_peak, ma_t_wave, window_len, value_to_align):
    factor_function = create_factor_function(0.25, 1.5, 0, window_len)
    quality_factor = 0
    if t_potential_start.size == 0 or t_potential_end.size == 0:
        return -1, -1, -1, factor_function, quality_factor
    else:
        if t_potential_start.size == 1:
            index = 0
            quality_factor = 1
        else:

            index, energy_blocks = choose_one_block(t_potential_start, t_potential_end, ma_peak, ma_t_wave, factor_function, value_to_align)
            all_energy = np.sum(energy_blocks)
            quality_factor = 1 - ((all_energy - energy_blocks[index]) / all_energy)


        signal = original_signal[t_potential_start[index]:t_potential_end[index]]
        signal_max_original = signal_filtered[t_potential_start[index]:t_potential_end[index]]
        peak_index = np.argmax(signal) + t_potential_start[index]
        peak_index_max_original = np.argmax(signal_max_original) + t_potential_start[index]
        if np.abs(peak_index_max_original - peak_index) > int(0.05 * fs):
            peak_index_max_original = peak_index
        return t_potential_start[index], peak_index_max_original, t_potential_end[index], factor_function, quality_factor


def choose_one_block(t_potential_start, t_potential_end, ma_peak, ma_t_wave, factor_function, value_to_align):
    max_value = 0
    max_index = 0
    energy_blocks = np.zeros(t_potential_start.size, dtype=float)
    block_len = np.zeros(t_potential_start.size, dtype=int)
    for index in range(min(t_potential_start.size, t_potential_end.size)):
        distance = np.abs(ma_peak[t_potential_start[index]:t_potential_end[index]] - ma_t_wave[t_potential_start[index]:t_potential_end[index]])
        block_len[index] = t_potential_end[index] - t_potential_start[index]
        t_potential_start_align_to_zero = t_potential_start - value_to_align
        t_potential_end_align_to_zero = t_potential_end - value_to_align
        current = np.sum(distance * factor_function[t_potential_start_align_to_zero[index]:t_potential_end_align_to_zero[index]])
        energy_blocks[index] = current
        #current2 = np.sum(distance)
        if max_value < current:
            max_index = index
            max_value = current
    #print(block_len)
    # for i in range(block_len.size):
    #     if i != max_index and block_len[i] / block_len[max_index] >= 1.25:
    #         max_index = i
    #print(energy_blocks)
    return max_index, energy_blocks


def create_factor_function(initial_value, middle_value, final_value, window_len):
    function = np.zeros(window_len, dtype=float)
    first_window_size = int(0.27*window_len - 1)# TODO
    if first_window_size == 1:
        first_window_size += 1
    m1 = (middle_value - initial_value) / (first_window_size - 1)
    m2 = (final_value - middle_value) / (window_len - first_window_size)
    for index in range(first_window_size):
        function[index] = initial_value + m1*index
    for index2 in range(first_window_size, window_len):
        #print(index2, first_window_size, window_len)
        function[index2] = middle_value * math.exp(2*m2*(index2 - first_window_size + 1))
        #function[index2] = middle_value + m2 * (index2 - first_window_size + 1)
    # print(function.size, function)
    return function


def union_block_of_interest(boi_diff, min_size_of_window_to_union, max_distance_of_windows_to_union):
    start_index_list = []
    end_index_list = []

    for index, value in enumerate(boi_diff):
        if value == 1:
            start_index_list.append(index)
        elif value == -1:
            end_index_list.append(index)
    for i in range(len(start_index_list) - 1):
        if ((end_index_list[i] - start_index_list[i] >= min_size_of_window_to_union) or (end_index_list[i+1] - start_index_list[i+1] >= min_size_of_window_to_union)) and (start_index_list[i+1] - end_index_list[i] <= max_distance_of_windows_to_union):
            boi_diff[start_index_list[i+1]] = 0
            boi_diff[end_index_list[i]] = 0
            #print(start_index_list[i+1],end_index_list[i] )
            #print('united')
            #i += 1
    return boi_diff




def t_peaks_annotations(ecg_original, chosen_ann, seg=0, all_seg=False):
    fs = ecg_original['fs']
    signal_len_in_time = ecg_original['signal_len']
    annotations_samples = []
    annotations_markers = []
    if chosen_ann == "real":
        if all_seg:
            for seg in range(ecg_original["num_of_segments"]):
                annotations_samples.extend(ecg_original["ann"][seg])
                annotations_markers.extend(ecg_original["ann_markers"][seg])
        else:
            annotations_samples = ecg_original["ann"][seg]
            annotations_markers = ecg_original["ann_markers"][seg]
    else:
        if all_seg:
            for seg in range(ecg_original["num_of_segments"]):
                annotations_samples.extend(ecg_original["our_ann"][seg])
                annotations_markers.extend(ecg_original["our_ann_markers"][seg])
        else:
            annotations_samples = ecg_original["our_ann"][seg]
            annotations_markers = ecg_original["our_ann_markers"][seg]
    len_ann = len(annotations_samples)
    t_annotations = np.zeros(len_ann, dtype=int)
    for index, marker in enumerate(annotations_markers):
        if marker == 't' and index != 0 and annotations_markers[index - 1] == '(':  # t_peak marker is 't'
            t_annotations = np.insert(t_annotations, 0, annotations_samples[index])

    t_annotations = t_annotations[t_annotations != 0]
    t_annotations = np.sort(t_annotations)
    t_annotations = t_annotations - seg * signal_len_in_time * fs
    return t_annotations

def comparison_t_peaks(t_peaks_real_annotations, t_peaks_our_annotations, fs, r_intervals_size, margin_mistake_in_sec=0.030):
    distance_from_real = np.zeros(t_peaks_real_annotations.size, dtype=int)
    success = 0
    margin_mistake = round(margin_mistake_in_sec*fs)

    for i in range(distance_from_real.size):
        for j in range(t_peaks_our_annotations.size):
            if t_peaks_our_annotations[j] != -1:
                distance = abs(t_peaks_real_annotations[i] - t_peaks_our_annotations[j])
            else:
                distance = margin_mistake + 1
            if distance <= margin_mistake:
                distance_from_real[i] = 1
                t_peaks_our_annotations[j] = -1
                success = success + 1
                break  # break the t_peaks_our_ann for
        # if distance_from_real[i] == 0:
        #    print(t_peaks_real_annotations[i] / fs)
    return success, distance_from_real.size





