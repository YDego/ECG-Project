import math
import wfdb
import processing_functions
from wfdb import processing
import numpy as np

def detection_qrs(ecg_original_copy):
    original_signal = ecg_original_copy["signal"]
    fs = ecg_original_copy["fs"]
    re_check_samples = round(0.2*fs)
    new_signal = processing_functions.band_pass_filter(8, 49, original_signal, fs)
    signal = abs(new_signal)**3
    ecg_original_copy["signal"] = signal
    ecg_original_copy["fft"], ecg_original_copy["frequency_bins"] = processing_functions.compute_fft(signal, fs)
    threshold = np.mean(signal)

    open_dots, closed_dots, all_dots = detection_qrs_aux_new(signal, threshold, 0.4, 0, False, fs)
    single_open_dots, single_closed_dots = check_for_singles_dots(open_dots, closed_dots, all_dots, fs)

    for dot in single_open_dots:
        if dot > len(signal) - re_check_samples:
            open_dots = np.delete(open_dots , np.where(open_dots == dot))
            all_dots = np.delete(all_dots, np.where(all_dots == dot))
            continue
        _, extra_close_dot, _ = detection_qrs_aux_new(signal[dot:dot+re_check_samples], threshold/2, 0.6, 1, True, fs)
        closed_dots = np.concatenate((extra_close_dot + dot, closed_dots))

    for dot in single_closed_dots:
        if dot < re_check_samples:
            closed_dots = np.delete(closed_dots, np.where(closed_dots == dot))
            all_dots = np.delete(all_dots, np.where(all_dots == dot))
            continue
        extra_open_dot, _, _ = detection_qrs_aux_new(signal[dot - re_check_samples:dot], threshold/2, 0.6, 0, True, fs)
        open_dots = np.concatenate((dot - extra_open_dot, open_dots))

    closed_dots = sorted(closed_dots)
    open_dots = sorted(open_dots)
    r_peaks_potential = wfdb.processing.find_local_peaks(original_signal, radius=round(0.05*fs))## todo
    r_peaks = find_r_peak(r_peaks_potential, open_dots, closed_dots, original_signal, fs)
    all_dots = sorted(np.concatenate((open_dots, r_peaks, closed_dots)))
    ecg_original_copy["our_ann"] = all_dots
    ecg_original_copy["our_ann_markers"] = np.zeros(len(all_dots), dtype=str)
    for index, dot in enumerate(all_dots):
        if dot in open_dots:
            ecg_original_copy["our_ann_markers"][index] = '('
        elif dot in closed_dots:
            ecg_original_copy["our_ann_markers"][index] = ')'
        else:
            ecg_original_copy["our_ann_markers"][index] = 'N'

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
    all_dots = np.concatenate((open_dots, closed_dots))
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
def r_peaks_annotations(ecg_original, chosen_ann):
    if chosen_ann == "real":
        real_annotations_samples = ecg_original["ann"]
        real_annotations_markers = ecg_original["ann_markers"]
    else:
        real_annotations_samples = ecg_original["our_ann"]
        real_annotations_markers = ecg_original["our_ann_markers"]

    r_peaks_real_annotations = np.zeros(len(real_annotations_samples), dtype=int)
    for index, marker in enumerate(real_annotations_markers):
        if marker == 'N': ## r_peak marker is 'N'
            r_peaks_real_annotations = np.insert(r_peaks_real_annotations, 0, real_annotations_samples[index])

    r_peaks_real_annotations = r_peaks_real_annotations[r_peaks_real_annotations != 0]
    r_peaks_real_annotations = np.sort(r_peaks_real_annotations)
    return r_peaks_real_annotations


### changed at 16.6.23 - 22:45
def comprasion_r_peaks(ecg_dict):
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
        if distance_from_real[index] <= 10:## 6 ms
            success = success + 1
    ecg_dict["r_peak_success"] = [success, number_of_dots]
    return ecg_dict


def check_for_singles_dots(open_dots, closed_dots, all_dots, fs):
    ## check that closed dot comes after open and there is not 2 open dot in a row

    impossible_margin = round(0.3*fs)
    list_of_single_open_dots = []
    list_of_single_closed_dots = []
    index_open, index_close = 0, 0
    while index_open < len(open_dots) and index_close < len(closed_dots):
        if closed_dots[index_close] - open_dots[index_open] < 0:
            list_of_single_closed_dots.append(closed_dots[index_close])
            closed_dots = np.delete(closed_dots, index_close)
        elif closed_dots[index_close] - open_dots[index_open] > impossible_margin:
            list_of_single_open_dots.append(open_dots[index_open])
            open_dots = np.delete(open_dots, index_open)

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

    return list_of_single_open_dots, list_of_single_closed_dots

#todo
def find_r_peak(r_peak_potential, q_peak, s_peak, original_signal, fs):
    r_peak_potential = wfdb.processing.find_local_peaks(original_signal, radius=round(0.05 * fs))
    r_peak_potential_minimum = wfdb.processing.find_local_peaks((-1)*original_signal, radius=round(0.05 * fs))
    r_peak = np.zeros(len(q_peak), dtype=int)
    for index, value in enumerate(q_peak):
        potential_r_peak_one_interval = {}
        for value_r in r_peak_potential:
            if q_peak[index] < value_r < s_peak[index]:
                potential_r_peak_one_interval[value_r] = original_signal[value_r]
            else:
                continue
        if len(potential_r_peak_one_interval) != 0:
            r_peak[index] = max(potential_r_peak_one_interval, key=potential_r_peak_one_interval.get)

        else:
            for value_r_minimum in r_peak_potential_minimum:
                if q_peak[index] < value_r_minimum < s_peak[index]:
                    potential_r_peak_one_interval[value_r_minimum] = original_signal[value_r_minimum]
                else:
                    continue
            if len(potential_r_peak_one_interval) != 0:
                r_peak[index] = min(potential_r_peak_one_interval, key=potential_r_peak_one_interval.get)

    return r_peak
