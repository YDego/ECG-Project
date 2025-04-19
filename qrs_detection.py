import math
import wfdb
import processing_functions
from wfdb import processing
import numpy as np
import copy


def detect_qrs(ecg_dict):
    ecg_dict_qrs_detected = copy.deepcopy(ecg_dict)
    for seg in range(ecg_dict['num_of_segments']):
        ecg_dict_qrs_detected = detect_qrs_single_segment(ecg_dict, seg)
    return ecg_dict_qrs_detected


def detect_qrs_single_segment(ecg_original_copy, seg):
    original_signal = ecg_original_copy["signal"][seg]
    fs = ecg_original_copy["fs"]
    re_check_samples = round(0.2*fs)
    new_signal = processing_functions.band_pass_filter(8, 49, original_signal, fs)
    #new_signal = processing_functions.wavelet_filter(original_signal)
    signal = abs(new_signal)**3
    threshold = np.mean(signal[round(0.1*fs): signal.shape[-1] - round(0.1*fs)])
    signal[0:round(0.1*fs) - 1] = threshold
    signal[signal.shape[-1] - round(0.1*fs): signal.shape[-1] - 1] = threshold
    ecg_original_copy["signal"][seg] = signal
    ecg_original_copy["fft"][seg], ecg_original_copy["frequency_bins"][seg] = processing_functions.compute_fft(new_signal, fs)
    threshold = np.mean(signal)

    open_dots, closed_dots, all_dots = detection_qrs_aux_new(signal, threshold, 0.4, 0, False, fs)
    single_open_dots, single_closed_dots, open_dots, closed_dots = check_for_singles_dots(open_dots, closed_dots, all_dots, fs)

    for dot in single_open_dots:
        if dot > len(signal) - re_check_samples:
            open_dots = np.delete(open_dots , np.where(open_dots == dot))
            all_dots = np.delete(all_dots, np.where(all_dots == dot))
            continue
        _, extra_close_dot, _ = detection_qrs_aux_new(signal[dot:dot+re_check_samples], threshold/2, 0.6, 1, True, fs)
        closed_dots = np.concatenate((extra_close_dot + dot, closed_dots), axis=None)

    for dot in single_closed_dots:
        if dot < re_check_samples:
            closed_dots = np.delete(closed_dots, np.where(closed_dots == dot))
            all_dots = np.delete(all_dots, np.where(all_dots == dot))
            continue
        extra_open_dot, _, _ = detection_qrs_aux_new(signal[dot - re_check_samples:dot], threshold/2, 0.6, 0, True, fs)
        open_dots = np.concatenate((dot - re_check_samples + extra_open_dot, open_dots), axis=None)

    closed_dots = sorted(closed_dots)
    open_dots = sorted(open_dots)
    r_peaks = find_r_peak(open_dots, closed_dots, new_signal, fs)
    all_dots = sorted(np.concatenate((open_dots, r_peaks, closed_dots)))

    ecg_original_copy["our_ann"].append(list(all_dots))
    ecg_original_copy["our_ann_markers"].append(list(np.zeros(len(all_dots), dtype=str)))
    for index, dot in enumerate(all_dots):
        if dot in open_dots:
            ecg_original_copy["our_ann_markers"][-1][index] = '('
        elif dot in closed_dots:
            ecg_original_copy["our_ann_markers"][-1][index] = ')'
        else:
            ecg_original_copy["our_ann_markers"][-1][index] = 'n'

    return ecg_original_copy


"""
def detection_qrs_aux(signal, threshold, margin_error, start_location, one_point, fs):
    location = start_location
    location_for_open = 0
    location_for_closed = 0
    MAX_NUM_PULSE = 5000
    #print(signal)
    times_repair_open = 0
    times_repair_closed = 0
    steps_to_check = round(0.03*fs)
    impossible_margin = round(0.2*fs)
    fix_error = 50
    open_dots = np.zeros(MAX_NUM_PULSE, dtype=int)
    closed_dots = np.zeros(MAX_NUM_PULSE, dtype=int)
    sus_dots = np.zeros(MAX_NUM_PULSE, dtype=int)
    for index, value in enumerate(signal):
        if (location % 2 == 0) and (index < signal.shape[-1] - steps_to_check) and (index > steps_to_check):
            times_repair_closed = 0
            if check_radius_open_dot(signal, index, threshold, steps_to_check, margin_error):
                if (location_for_open != 0) and abs(index - open_dots[location_for_open - 1]) < impossible_margin:  ## open dot cant be in radius of open dot so fast
                    times_repair_open = times_repair_open + 1
                    open_dots[location_for_open - 1] = open_dots[location_for_open-1] + ((-index + open_dots[location_for_open - 1])/(2**times_repair_open))
                #elif (location_for_open != 0) and abs(index - open_dots[location_for_open - 1]) < impossible_margin:
                    #continue
                #elif (location_for_closed != 0) and ((index - closed_dots[location_for_closed-1]) < 2*impossible_margin):## open dot cant be after closed dot so fast
                    #sus_dots[location] = index
                else:
                    open_dots[location_for_open] = index
                    location_for_open = location_for_open + 1
                    location = location + 1
                    if one_point:
                        break
            else:
                continue

        elif (location % 2 == 1) and (index < signal.shape[-1] - steps_to_check) and (index > steps_to_check):
            times_repair_open = 0
            index_from_end = signal.shape[-1] - index - 1
            if check_radius_closed_dot(signal, index_from_end, threshold,steps_to_check, margin_error):
                if ((location_for_closed != 0) and abs(index_from_end - closed_dots[location_for_closed-1]) < impossible_margin):
                    times_repair_closed = times_repair_closed + 1
                    closed_dots[location_for_closed - 1] = closed_dots[location_for_closed - 1] - ((-index_from_end + closed_dots[location_for_closed - 1]) / (2**times_repair_closed))
                #elif ((location_for_closed != 0) and abs(index_from_end - closed_dots[location_for_closed-1]) < impossible_margin):
                    #continue
                else:
                    closed_dots[location_for_closed] = index_from_end
                    location_for_closed = location_for_closed + 1
                    location = location + 1
                    if one_point:
                        break
                continue

    sus_dots = sus_dots[sus_dots != 0]
    open_dots = open_dots[open_dots != 0]
    closed_dots = closed_dots[closed_dots != 0]
    closed_dots = np.flip(closed_dots)
    all_dots = np.concatenate((open_dots, closed_dots))
    all_dots = sorted(all_dots)
    return open_dots, closed_dots, all_dots
"""


def detection_qrs_aux_new(signal, threshold, margin_error, start_from, one_point, fs):
    location = start_from
    location_for_open = 0
    location_for_closed = 0
    max_num_dots = (fs * 100)
    # print(signal)
    times_repair_open = 0
    times_repair_closed = 0
    steps_to_check = round(0.03*fs) ## 15
    impossible_margin = round(0.2*fs) ## 100
    fix_error = round(0.1*fs) ## 50
    max_of_repair = round(0.02*fs) ## 10
    open_dots = np.zeros(max_num_dots, dtype=int)
    closed_dots = np.zeros(max_num_dots, dtype=int)
    repair_dots = []

    if location == 0:
        for index, value in enumerate(signal):
            if (index < signal.shape[-1] - steps_to_check) and (index > steps_to_check):
                if check_radius_open_dot(signal, index, threshold, steps_to_check, margin_error):
                    if times_repair_open <= max_of_repair and (location_for_open != 0) and abs(index - open_dots[location_for_open - 1]) < fix_error: ## open dot cant be in radius of open dot so fast
                        if times_repair_open == 0:
                            repair_dots.append(open_dots[location_for_open - 1])
                        times_repair_open = times_repair_open + 1
                        repair_dots.append(index)
                        open_dots[location_for_open - 1] = np.mean(repair_dots)
                    elif (location_for_open != 0) and abs(index - open_dots[location_for_open - 1]) < impossible_margin + fix_error:
                        continue
                    else:
                        open_dots[location_for_open] = index
                        location_for_open = location_for_open + 1
                        times_repair_open = 0
                        location = location + 1
                        repair_dots = []
                        if one_point:
                            break
                else:
                    continue
    if not one_point:
        location = 1

    repair_dots = []
    if location == 1:
        for index, value in enumerate(signal):
            index_from_end = signal.shape[-1] - index
            if (index_from_end < signal.shape[-1] - steps_to_check) and (index_from_end > steps_to_check):
                if check_radius_closed_dot(signal, index_from_end, threshold, steps_to_check, margin_error):
                    if times_repair_closed <= max_of_repair and (location_for_closed != 0) and abs(index_from_end - closed_dots[location_for_closed - 1]) < fix_error: ## open dot cant be in radius of open dot so fast
                        if times_repair_closed == 0:
                            repair_dots.append(closed_dots[location_for_closed - 1])
                        times_repair_closed = times_repair_closed + 1
                        repair_dots.append(index_from_end)
                        closed_dots[location_for_closed - 1] = np.mean(repair_dots)
                    elif (location_for_closed != 0) and abs(index_from_end - closed_dots[location_for_closed - 1]) < impossible_margin + fix_error:
                        continue
                    else:
                        closed_dots[location_for_closed] = index_from_end
                        location_for_closed = location_for_closed + 1
                        times_repair_closed = 0
                        location = location + 1
                        repair_dots = []
                        if one_point:
                            break
                else:
                    continue
    #sus_dots = sus_dots[sus_dots != 0]
    open_dots = open_dots[open_dots != 0]
    closed_dots = closed_dots[closed_dots != 0]
    closed_dots = np.flip(closed_dots)
    all_dots = np.concatenate((open_dots, closed_dots), axis=None)
    all_dots = sorted(all_dots)
    return open_dots, closed_dots, all_dots


def check_radius_open_dot(signal, index, threshold, distance, margin_error):
    flag_forward = True
    false_dot_forward = 0
    margin_of_error = margin_error
    original_distance = distance

    #while distance > 0:
        #if signal[index+distance] < threshold:
            #false_dot_forward = false_dot_forward + 1
        #distance = distance - 1

    #if false_dot_forward > margin_of_error*original_distance:
    if threshold > np.mean(signal[index:index+distance]):#new line
        flag_forward = False
    #distance = distance - 1
    flag_backward = True
    false_dot_backward = 0

    #while distance + 1 > - original_distance:
        #if signal[index+distance] > threshold:
            #false_dot_backward = false_dot_backward + 1
        #distance = distance - 1

    #if false_dot_backward > margin_of_error*original_distance:
    if threshold < np.mean(signal[index-distance:index]):
        flag_backward = False
    return flag_backward and flag_forward


def check_radius_closed_dot(signal, index, threshold, distance, margin_error):
    flag_forward = True

    if threshold < np.mean(signal[index:index+distance]):#new
        flag_forward = False
    flag_backward = True

    if threshold > np.mean(signal[index-distance:index]):#new
        flag_backward = False
    return flag_backward and flag_forward


### changed at 16.6.23 - 22:45
def r_peaks_annotations(ecg_original, chosen_ann, seg=0, all_seg=False):
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

    r_peaks_real_annotations = np.zeros(len(annotations_samples), dtype=int)
    for index, marker in enumerate(annotations_markers):
        if marker == 'N' or marker == 'n': ## r_peak marker is 'N'
            r_peaks_real_annotations = np.insert(r_peaks_real_annotations, 0, annotations_samples[index]) ## check with my version todo -> annotations_sampels , was markers

    r_peaks_real_annotations = r_peaks_real_annotations[r_peaks_real_annotations != 0]
    r_peaks_real_annotations = np.sort(r_peaks_real_annotations)
    r_peaks_real_annotations = r_peaks_real_annotations - seg * signal_len_in_time * fs
    return r_peaks_real_annotations


### changed at 16.6.23 - 22:45

"""
def comparison_r_peaks(ecg_dict):
    r_peaks_real_annotations = r_peaks_annotations(ecg_dict, 'real')
    print(r_peaks_real_annotations)
    r_peaks_our_annotations = r_peaks_annotations(ecg_dict, 'our')
    print(r_peaks_our_annotations)
    len_iter = min(len(r_peaks_real_annotations), len(r_peaks_our_annotations))
    location_to_start = 0
    min_distance = math.inf
    for index in range(len_iter):##need to check about len iter
        distance = abs(r_peaks_real_annotations[0] - r_peaks_our_annotations[index])
        if distance < min_distance:
            min_distance = distance
            location_to_start = index
    distance_from_real = np.zeros(len(r_peaks_real_annotations), dtype=int)
    r_peaks_our_annotations = np.delete(r_peaks_our_annotations, np.arange(0, location_to_start, 1))
    len_iter = min(len(r_peaks_real_annotations), len(r_peaks_our_annotations))
    success = 0
    number_of_dots = 0
    for index in range(len_iter):
        distance_from_real[index] = abs(r_peaks_real_annotations[index] - r_peaks_our_annotations[index])
        number_of_dots = number_of_dots + 1
        if distance_from_real[index] <= 15:## 6 ms
            success = success + 1
    ecg_dict["r_peak_success"] = [success, number_of_dots]
    return ecg_dict
"""


##changed at 25.6 01:18
def comparison_r_peaks(ecg_dict):
    ecg_dict_r_compared = copy.deepcopy(ecg_dict)
    ecg_dict_r_compared["r_peak_success"] = []
    for seg in range(ecg_dict['num_of_segments']):
        ecg_dict_r_compared = comparison_r_peaks_single_segment(ecg_dict_r_compared, seg)
    return ecg_dict_r_compared


def comparison_r_peaks_single_segment(ecg_dict, seg):
    r_peaks_real_annotations = r_peaks_annotations(ecg_dict, 'real', seg)
    # print(r_peaks_real_annotations)
    r_peaks_our_annotations = r_peaks_annotations(ecg_dict, 'our', seg)
    # print(r_peaks_our_annotations)
    len_of_real_r_peaks = len(r_peaks_real_annotations)
    i = 0
    min_distance = math.inf
    distance_from_real = np.zeros(min(len(r_peaks_real_annotations), len(r_peaks_our_annotations)), dtype=int)
    success = 0
    number_of_dots = 0
    for i in range(min(len(r_peaks_real_annotations), len(r_peaks_our_annotations))):
        min_distance = math.inf
        index_our_to_delete = -1
        index_real_to_delete = -1
        for index_real, value_real in enumerate(r_peaks_real_annotations):
            if min_distance < 25:
                break
            if value_real == 0:
                continue
            for index_our, value_our in enumerate(r_peaks_our_annotations):
                distance = abs(r_peaks_real_annotations[index_real] - r_peaks_our_annotations[index_our])
                if min_distance < 25:
                    break
                if value_our == 0:
                    continue
                if distance < min_distance:
                    min_distance = distance
                    index_real_to_delete = index_real
                    index_our_to_delete = index_our
        distance_from_real[i] = min_distance
        r_peaks_our_annotations[index_our_to_delete] = 0
        r_peaks_real_annotations[index_real_to_delete] = 0

    if len(r_peaks_real_annotations[r_peaks_real_annotations != 0]) != 0:
        number_of_dots = number_of_dots + len(r_peaks_real_annotations[r_peaks_real_annotations != 0])

    for index in range(len(distance_from_real)):
        number_of_dots = number_of_dots + 1
        if distance_from_real[index] <= round(0.025 * ecg_dict["fs"]):  ##
            success = success + 1
    ecg_dict["r_peak_success"].append([success, number_of_dots])
    return ecg_dict


def check_for_singles_dots(open_dots, closed_dots, all_dots, fs):
    ## check that closed dot comes after open and there is not 2 open dot in a row
    open_dots_for_delete = open_dots.copy()
    closed_dots_for_delete = closed_dots.copy()
    impossible_margin_max = round(0.3*fs)
    impossible_margin_min = round(0.03*fs)
    list_of_single_open_dots = []
    list_of_single_closed_dots = []
    index_open, index_close = 0, 0
    while index_open < len(open_dots) and index_close < len(closed_dots):
        if closed_dots[index_close] - open_dots[index_open] < 0:
            list_of_single_closed_dots.append(closed_dots[index_close])
            closed_dots = np.delete(closed_dots, index_close)
        elif closed_dots[index_close] - open_dots[index_open] > impossible_margin_max:
            list_of_single_open_dots.append(open_dots[index_open])
            open_dots = np.delete(open_dots, index_open)
        elif closed_dots[index_close] - open_dots[index_open] < impossible_margin_min:
            closed_dots_for_delete = np.delete(closed_dots_for_delete, np.where(closed_dots_for_delete == closed_dots[index_close]))
            open_dots_for_delete = np.delete(open_dots_for_delete, np.where(open_dots_for_delete == open_dots[index_open]))
            index_close = index_close + 1
            index_open = index_open + 1
        else:
            index_close = index_close + 1
            index_open = index_open + 1
    if len(open_dots) < len(closed_dots):
        while index_open < len(closed_dots):
            list_of_single_closed_dots.append(closed_dots[index_open])
            index_open = index_open + 1
    elif len(open_dots) > len(closed_dots):
        while index_open < len(open_dots):
            list_of_single_open_dots.append(open_dots[index_open])
            index_open = index_open + 1

    return list_of_single_open_dots, list_of_single_closed_dots , open_dots_for_delete, closed_dots_for_delete


#todo
def find_r_peak(q_peak, s_peak, original_signal, fs):
    r_peak_potential = wfdb.processing.find_local_peaks(original_signal, radius=round(0.05 * fs))
    r_peak_potential_minimum = wfdb.processing.find_local_peaks((-1)*original_signal, radius=round(0.05 * fs))
    r_peak = np.zeros(len(q_peak), dtype=int)
    index_next_r, index_next_r_min = 0, 0
    for index in range(0, len(q_peak), 1):
        if index >= len(s_peak):
            continue
        potential_r_peak_one_interval = {}
        index_r = 0
        while index_r < len(r_peak_potential): ### change 28/02 speed change only
            if index_next_r != 0:
                index_r, index_next_r = index_next_r, 0
            value_r = r_peak_potential[index_r]
            q_peak_ = q_peak[index]  ## debug
            s_peak_ = s_peak[index]  ## debug
            if s_peak[index] < q_peak[index]:
                s_peak = np.delete(s_peak, index)
                continue
            if q_peak[index] < value_r < s_peak[index]:
                potential_r_peak_one_interval[value_r] = original_signal[value_r]
                index_r += 1
                if index == 427:
                    print('here')
                    potential_r_peak_one_interval[value_r] = original_signal[value_r]
                    index_r += 1
            else:
                if value_r > s_peak[index]: ## r peak not before s peak. 28/02/25
                    index_next_r = index_r
                    break
                else:
                    index_r += 1
                    continue
        if len(potential_r_peak_one_interval) != 0:
            r_peak[index] = max(potential_r_peak_one_interval, key=potential_r_peak_one_interval.get)
        else:
            index_r_min = 0
            while index_r_min < len(r_peak_potential_minimum):  ### change 28/02 speed change only
                if index_next_r_min != 0:
                    index_r_min, index_next_r_min = index_next_r_min, 0
                value_r_min = r_peak_potential_minimum[index_r_min]
                if q_peak[index] < value_r_min < s_peak[index]:
                    potential_r_peak_one_interval[value_r_min] = original_signal[value_r_min]
                    index_r_min += 1
                    if index == 428:
                        print('here')
                        potential_r_peak_one_interval[value_r_min] = original_signal[value_r_min]
                        index_r_min += 1
                else:
                    if value_r_min > s_peak[index]:  ## r peak not before s peak. 28/02/25
                        index_next_r_min = index_r_min
                        break
                    else:
                        index_r_min += 1
                        continue
            if len(potential_r_peak_one_interval) != 0:
                r_peak[index] = min(potential_r_peak_one_interval, key=potential_r_peak_one_interval.get)

    return r_peak


def find_q_s_ann(ecg_original_copy, seg=0, findQann = False , findSann = False, realLabels = True, all_seg=False):
    fs = ecg_original_copy['fs']
    signal_len_in_time = ecg_original_copy['signal_len']
    ann = []
    ann_markers = []
    if realLabels:
        if all_seg:
            for seg in range(ecg_original_copy["num_of_segments"]):
                ann.extend(ecg_original_copy["ann"][seg])
                ann_markers.extend(ecg_original_copy["ann_markers"][seg])
        else:
            ann = ecg_original_copy["ann"][seg]
            ann_markers = ecg_original_copy["ann_markers"][seg]
    else:
        if all_seg:
            for seg in range(ecg_original_copy["num_of_segments"]):
                ann.extend(ecg_original_copy["our_ann"][seg])
                ann_markers.extend(ecg_original_copy["our_ann_markers"][seg])
        else:
            ann = ecg_original_copy["our_ann"][seg]
            ann_markers = ecg_original_copy["our_ann_markers"][seg]

    if findQann:
        q_ann = np.zeros(len(ann), dtype=int)
        q_ann_size = 0
        for index in range(0, len(ann_markers) - 1):
            if ann_markers[index] == '(' and (ann_markers[index + 1] == 'N' or ann_markers[index + 1] == 'n'):
                q_ann[q_ann_size] = ann[index]
                q_ann_size = q_ann_size + 1
        q_ann = q_ann[q_ann != 0]
    else:
        q_ann = -1

    if findSann:
        s_ann = np.zeros(len(ann), dtype=int)
        s_ann_size = 0
        for index in range(1, len(ann_markers)):
            if ann_markers[index] == ')' and (ann_markers[index - 1] == 'N' or ann_markers[index - 1] == 'n'):
                s_ann[s_ann_size] = ann[index]
                s_ann_size = s_ann_size + 1
        s_ann = s_ann[s_ann != 0]
    else:
        s_ann = -1

    q_ann = q_ann - seg * signal_len_in_time * fs
    s_ann = s_ann - seg * signal_len_in_time * fs
    return q_ann, s_ann

