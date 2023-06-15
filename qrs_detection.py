import math
import wfdb
from wfdb import processing
import numpy as np

def detection_qrs(original_signal ,signal,threshold):
    re_check_samples = 100
    impossible_margin = 150
    open_dots, closed_dots, all_dots = detection_qrs_aux_new(signal, threshold, 0.4, 0, False)
    single_open_dots, single_closed_dots = check_for_singles_dots(open_dots, closed_dots, all_dots)


    for dot in single_open_dots:
        if dot > len(signal) - re_check_samples:
            open_dots = np.delete(open_dots , np.where(open_dots == dot))
            all_dots = np.delete(all_dots, np.where(all_dots == dot))
            continue
        #elif np.any(open_dots[abs(open_dots - dot) < impossible_margin]):
            #continue
        _, extra_close_dot, _ = detection_qrs_aux_new(signal[dot:dot+re_check_samples], threshold/2, 0.6, 1, True)
        closed_dots = np.concatenate((extra_close_dot + dot, closed_dots))

    for dot in single_closed_dots:
        if dot < re_check_samples:
            closed_dots = np.delete(closed_dots, np.where(closed_dots == dot))
            all_dots = np.delete(all_dots, np.where(all_dots == dot))
            continue
        #elif np.any(closed_dots[abs(closed_dots - dot) < impossible_margin]):
            #continue
        extra_open_dot, _, _ = detection_qrs_aux_new(signal[dot - re_check_samples:dot], threshold/2, 0.6, 0, True)
        open_dots = np.concatenate((dot - extra_open_dot, open_dots))

    closed_dots = sorted(closed_dots)
    open_dots = sorted(open_dots)
    R_peaks_potential = wfdb.processing.find_local_peaks(original_signal, radius=25)
    r_peaks = find_r_peak(R_peaks_potential, open_dots, closed_dots , original_signal)
    all_dots = sorted(np.concatenate((open_dots, r_peaks, closed_dots)))

    return open_dots, r_peaks, closed_dots, all_dots


def detection_qrs_aux(signal, threshold, margin_error, start_location, one_point):
    location = start_location
    location_for_open = 0
    location_for_closed = 0
    MAX_NUM_PULSE = 5000
    #print(signal)
    times_repair_open = 0
    times_repair_closed = 0
    steps_to_check = 15
    impossible_margin = 100
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








def detection_qrs_aux_new(signal, threshold, margin_error, start_from, one_point):
    location = start_from
    location_for_open = 0
    location_for_closed = 0
    MAX_NUM_PULSE = 5000
    # print(signal)
    times_repair_open = 0
    times_repair_closed = 0
    steps_to_check = 15
    impossible_margin = 100
    fix_error = 50
    max_of_repair = 10
    open_dots = np.zeros(MAX_NUM_PULSE, dtype=int)
    closed_dots = np.zeros(MAX_NUM_PULSE, dtype=int)
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

def check_radius_closed_dot(signal, index ,threshold, distance, margin_error):
    flag_forward = True
    #false_dot_forward = 0
    #margin_of_error = margin_error
    #orignal_distance = distance
    #while distance > 0:
        #if signal[index + distance] > threshold:
            #false_dot_forward = false_dot_forward + 1
        #distance = distance - 1
    #if false_dot_forward > margin_of_error * orignal_distance:
    if threshold < np.mean(signal[index:index+distance]):#new
        flag_forward = False
    #istance = distance - 1
    flag_backward = True
    false_dot_backward = 0
    #while distance + 1 > - orignal_distance:
        #if signal[index + distance] < threshold:
            #false_dot_backward = false_dot_backward + 1
        #distance = distance - 1
    #if false_dot_backward > margin_of_error * orignal_distance:
    if threshold > np.mean(signal[index-distance:index]):#new
        flag_backward = False
    return flag_backward and flag_forward



def annotation_for_q_and_s(annotation_sample):
    location = 0
    only_q_s = np.zeros(len(annotation_sample), dtype=int)
    for index, value in enumerate(annotation_sample):
        if index % 9 == 0 or index % 9 == 1 or index % 9 == 2:
            only_q_s[index] = value
    only_q_s = only_q_s[only_q_s != 0]
    return only_q_s



def distance_from_real_dot(qrs_sample , our_samples):
    location_to_start = 0
    len_q_s_sample = len(qrs_sample)
    len_iter = min(len_q_s_sample, len(our_samples))
    min_distance = math.inf
    for index in range(len_iter):##need to check about len iter TODO
        distance = abs(qrs_sample[0] - our_samples[index])
        if distance < min_distance:
            min_distance = distance
            location_to_start = index

    distance_from_real = np.zeros(len_q_s_sample, dtype=int)
    our_samples = np.delete(our_samples, np.arange(0, location_to_start, 1))
    len_iter = min(len_q_s_sample, len(our_samples))
    for index in range(len_iter):
        distance_from_real[index] = qrs_sample[index] - our_samples[index]
    return distance_from_real


def check_for_singles_dots(open_dots
                    ,closed_dots,
                           all_dots):
    ##TODO check that closed dot comes after open and there is not 2 open dot in a row

    impossible_margin = 150
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


def find_r_peak(r_peak_potential , q_peak , s_peak , original_signal):
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
    return r_peak

