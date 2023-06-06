import numpy as np
import pywt


def detection_qrs(signal, threshold):
    location = 0
    MAX_NUM_PULSE = 5000
    #print(signal)
    steps_to_check = 10
    open_dots = np.zeros(MAX_NUM_PULSE, dtype=int)
    closed_dots = np.zeros(MAX_NUM_PULSE, dtype=int)
    for index, value in enumerate(signal):
        if (location % 2 == 0) and (index < signal.shape[-1] - steps_to_check) and (index > steps_to_check):
            if check_radius_open_dot(signal, index, threshold,steps_to_check):##try to find the first dot that greater then zero
                open_dots[location] = index
                location = location + 1
            else:
                continue

        elif (location % 2 == 1) and (index < signal.shape[-1] - steps_to_check) and (index > steps_to_check):
            if check_radius_closed_dot(signal, index, threshold,steps_to_check):
                closed_dots[location] = index
                location = location + 1
                continue
    open_dots = open_dots[open_dots != 0]
    closed_dots = closed_dots[closed_dots != 0]
    return open_dots, closed_dots


def check_radius_open_dot(signal, index, threshold, distance):
    flag_forward = True
    false_dot_forward = 0
    margin_of_error = 0.4
    original_distance = distance
    while distance > 0:
        if signal[index+distance] < threshold:
            false_dot_forward = false_dot_forward + 1
        distance = distance - 1
    if false_dot_forward > margin_of_error*original_distance:
        flag_forward = False
    distance = distance - 1
    flag_backward = True
    false_dot_backward = 0
    while distance + 1 > - original_distance:
        if signal[index+distance] > threshold:
            false_dot_backward = false_dot_backward + 1
        distance = distance - 1
    if false_dot_backward > margin_of_error*original_distance:
        flag_backward = False
    return flag_backward and flag_forward


def check_radius_closed_dot(signal, index, threshold, distance):
    flag_forward = True
    false_dot_forward = 0
    margin_of_error = 0.4
    orignal_distance = distance
    while distance > 0:
        if signal[index + distance] > threshold:
            false_dot_forward = false_dot_forward + 1
        distance = distance - 1
    if false_dot_forward > margin_of_error * orignal_distance:
        flag_forward = False
    distance = distance - 1
    flag_backward = True
    false_dot_backward = 0
    while distance + 1 > - orignal_distance:
        if signal[index + distance] < threshold:
            false_dot_backward = false_dot_backward + 1
        distance = distance - 1
    if false_dot_backward > margin_of_error * orignal_distance:
        flag_backward = False
    return flag_backward and flag_forward


def wavelet_filter(signal):
    wavelet = pywt.Wavelet('sym4')
    # levdec = min(pywt.dwt_max_level(signal.shape[-1], wavelet.dec_len), 6)
    Ca4, Cd4, Cd3, Cd2, Cd1 = pywt.wavedec(signal, wavelet=wavelet, level=4)

    Ca4, Cd2, Cd1 = np.zeros(Ca4.shape[-1]),np.zeros(Cd2.shape[-1]),np.zeros(Cd1.shape[-1])
    fitered_signal = pywt.waverec([Ca4, Cd4, Cd3, Cd2, Cd1], wavelet)
    return fitered_signal