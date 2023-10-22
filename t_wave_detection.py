import numpy as np
import qrs_detection
import plot_manager as pm
import math


# page 9-10

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
    if w_size % 2 != 1:
        print("moving average wrong, window size is not odd")
        exit(1)
    half_window = int((w_size - 1) / 2)
    for i in range(half_window, n - half_window):
        low_index = i - half_window
        high_index = i + half_window
        # Calculate the average of current window
        window_average = round(np.sum(arr[low_index:high_index + 1]) / w_size, 5)
        moving_averages[i] = window_average
    return moving_averages


def qrs_removal(ecg_original_copy, q_ann, s_ann):
    # return signal without QRS complex (return only the signal without all the dict)
    signal = ecg_original_copy["original_signal"]
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


def thresholding(ecg_copy, boi, r_peaks, w1_size=0.070, k=1):
    fs = ecg_copy['fs']
    signal = ecg_copy["original_signal"]
    d_max = 0.800  # 800 ms
    d_min = 0.170  # 170 ms
    # pm.plot_2_signals(signal, boi, fs)
    for i in range(0, r_peaks.size - 1):
        rr_interval = r_peaks[i + 1] - r_peaks[i]
        rt_min = int(np.ceil(d_min * rr_interval))
        rt_max = int(np.ceil(d_max * rr_interval))
        boi[r_peaks[i]: r_peaks[i] + int(rt_min)] = 0
        boi[r_peaks[i] + int(rt_max): r_peaks[i + 1]] = 0

    new_boi = np.zeros(len(boi), dtype=int)
    thr = np.floor(w1_size * fs * k)
    count = 0
    for i, x in enumerate(boi):
        if x == 1:
            count += 1
        elif count > int(thr):
            new_boi[i - count:i] = 1
            count = 0
        else:
            count = 0
    # pm.plot_2_signals(signal, new_boi, fs)
    return new_boi


def find_real_blocks(ecg_copy, signal_filtered, r_peaks, boi):
    t_potential_start_potential_end = np.diff(boi)  # 4999 size
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
        t_start = np.zeros(r_peaks.size - 1, dtype=int)
        t_end = np.zeros(r_peaks.size - 1, dtype=int)
        t_peak = np.zeros(r_peaks.size - 1, dtype=int)
        for index in range(0, r_peaks.size - 1):
            t_potential_start_one_interval = t_potential_start[r_peaks[index] < t_potential_start]
            t_potential_start_one_interval = t_potential_start_one_interval[
                t_potential_start_one_interval < r_peaks[index + 1]]
            t_potential_end_one_interval = t_potential_end[r_peaks[index] < t_potential_end]
            t_potential_end_one_interval = t_potential_end_one_interval[
                t_potential_end_one_interval < r_peaks[index + 1]]
            t_start[index], t_peak[index], t_end[index] = detected_real_block_of_interest(ecg_copy, signal_filtered,
                                                                                          t_potential_start_one_interval,
                                                                                          t_potential_end_one_interval)

        return t_start[t_start != 0], t_peak[t_peak != 0], t_end[t_end != 0]

#                               find t_peak over ****one interval**** , return also start and end of the block interest


#                                             inner function
def detected_real_block_of_interest(ecg_copy, signal_filtered, t_potential_start, t_potential_end):
    if t_potential_start.size == 0:
        return 0, 0, 0
    else:
        signal = ecg_copy['original_signal'][t_potential_start[0]:t_potential_end[0]]
        signal_max_original = signal_filtered[t_potential_start[0]:t_potential_end[0]]
        peak_index = np.argmax(signal) + t_potential_start[0]
        peak_index_max_original = np.argmax(signal_max_original) + t_potential_start[0]
        if np.abs(peak_index_max_original - peak_index) > int(0.05*ecg_copy['fs']):
            peak_index_max_original = peak_index
        return t_potential_start[0], peak_index_max_original, t_potential_end[0]


def compute_k_factor(r_peaks, fs):
    if r_peaks.size <= 1:
        return 1
    rr_intervals = np.diff(r_peaks)
    k = np.sum(rr_intervals) / (fs * rr_intervals.size)
    return k


def t_peaks_annotations(ecg_original, chosen_ann):
    if chosen_ann == "real":
        annotations_samples = ecg_original["ann"]
        annotations_markers = ecg_original["ann_markers"]
    else:
        annotations_samples = ecg_original["our_ann"]
        annotations_markers = ecg_original["our_ann_markers"]

    t_annotations = np.zeros(len(annotations_samples), dtype=int)
    for index, marker in enumerate(annotations_markers):
        if marker == 't' or marker == 'T':  # t_peak marker is 't'
            t_annotations = np.insert(t_annotations, 0, annotations_samples[index])

    t_annotations = t_annotations[t_annotations != 0]
    t_annotations = np.sort(t_annotations)
    return t_annotations


def comparison_t_peaks(t_peaks_real_annotations, t_peaks_our_annotations, fs):

    distance_from_real = np.zeros(min(len(t_peaks_real_annotations), len(t_peaks_our_annotations)), dtype=int)
    success = 0
    number_of_dots = 0
    for i in range(min(len(t_peaks_real_annotations), len(t_peaks_our_annotations))):
        min_distance = math.inf
        index_our_to_delete = -1
        index_real_to_delete = -1
        for index_real, value_real in enumerate(t_peaks_real_annotations):
            if min_distance < 25:
                break
            if value_real == 0:
                continue
            for index_our, value_our in enumerate(t_peaks_our_annotations):
                distance = abs(t_peaks_real_annotations[index_real] - t_peaks_our_annotations[index_our])
                if min_distance < 25:
                    break
                if value_our == 0:
                    continue
                if distance < min_distance:
                    min_distance = distance
                    index_real_to_delete = index_real
                    index_our_to_delete = index_our
        distance_from_real[i] = min_distance
        t_peaks_our_annotations[index_our_to_delete] = 0
        t_peaks_real_annotations[index_real_to_delete] = 0

    if len(t_peaks_real_annotations[t_peaks_real_annotations != 0]) != 0:
        number_of_dots = number_of_dots + len(t_peaks_real_annotations[t_peaks_real_annotations != 0])

    for index in range(len(distance_from_real)):
        number_of_dots = number_of_dots + 1
        if distance_from_real[index] <= round(0.05*fs):##
            success = success + 1

    return success, number_of_dots






