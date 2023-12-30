import numpy as np
import t_wave_detection
import time
import copy
import scipy
import qrs_detection as qrs

def main_p_peak_detection(ecg_dict_original, w1_size, signal_len_in_time, which_r_ann, real_q_s_ann=False):
    fs = ecg_dict_original['fs']
    ecg_dict_copy = copy.deepcopy(ecg_dict_original)
    p_peak_normal_all_seg_list = []
    for seg in range(0, ecg_dict_copy['num_of_segments']):
        signal = ecg_dict_copy['original_signal'][seg]
        b, a = scipy.signal.butter(2, [0.5, 12], btype='bandpass', output='ba', fs=fs)
        signal = scipy.signal.filtfilt(b, a, signal)
        q_ann, s_ann = qrs.find_q_s_ann(ecg_dict_original, seg, True, True, realLabels=real_q_s_ann)
        if q_ann.size == 0 or s_ann.size == 0:
            continue
        p_real_peaks = p_peaks_annotations(ecg_dict_original, 'real', seg)
        r_peaks = qrs.r_peaks_annotations(ecg_dict_original, which_r_ann, seg)
        #b2, a2 = scipy.signal.butter(2, 0.67, btype='highpass', output='ba', fs=fs)
        #signal_without_dc = scipy.signal.filtfilt(b2, a2, ecg_dict_copy['original_signal'][seg])
        signal_without_dc = ecg_dict_original['original_signal'][seg]
        #pvc_beats = p_wave_detection.pvc_detection(signal_without_dc, r_peaks, q_ann, s_ann, fs)
        signal_without_qrs = t_wave_detection.qrs_removal(signal, seg, q_ann, s_ann)
        p_peaks_united, p_peak_normal, p_peak_low = p_peak_detection(signal_without_qrs, fs, w1_size, 1, r_peaks, signal_without_qrs, 0.999, 0.7)
        #p_wave_detection.atrial_fib_detection(r_peaks, signal, p_peaks_united, signal_without_qrs)
        # pm.plot_signal_with_dots2(signal_without_dc, p_real_peaks, p_peaks_united, fs, 'original signal', 'p_real_peaks', 'our p peaks', i, seg, signal_len_in_time)
        p_peak_normal_all_seg_list.append(p_peak_normal + seg * signal_len_in_time * fs)
    p_peak_normal_all_seg_np = np.array(p_peak_normal_all_seg_list[0])
    return p_peak_normal_all_seg_np







def pvc_detection(signal, r_peaks, q_peaks, s_peaks, fs):
    r_peaks_size = r_peaks.size
    pvc_beats = np.zeros(r_peaks_size, dtype=bool)
    auc_beats = np.zeros(r_peaks_size, dtype=float)  # auc = area under qrs curve
    for index in range(r_peaks_size):
        #signal_seg = signal[q_peaks[index]:s_peaks[index]]
        signal_seg = signal[(r_peaks[index] - int(0.050 * fs)):(r_peaks[index] + int(0.050 * fs))]
        auc_beats[index] = np.trapz(np.abs(signal_seg), dx=1/fs) ## maybe shoulde take off the min value from all the array
    median = np.median(auc_beats)
    print(auc_beats)
    for index in range(r_peaks_size):
        if auc_beats[index] > 1.3 * median:
            pvc_beats[index] = True
            print(r_peaks[index] / fs)
            time.sleep(10)

    return pvc_beats


def atrial_fib_detection(r_peaks, signal, p_peaks, signal_without_qrs):
    rr_intervals = np.diff(r_peaks)
    #print(rr_intervals)
    #median_interval = np.median(rr_intervals)
    #print(median_interval)
    for index in range(0, rr_intervals.size - 2):  # sliding window of 3 intervals
        count = 0
        for slide in range(index, index + 3, 1):
            median_interval = np.median(rr_intervals[slide:slide+3])
            if np.abs(rr_intervals[slide] - median_interval) > median_interval * 0.5:
                count += 1
                print('irregular 1')
        if count == 2:
            print('irregular_beats')
    return






def p_peak_detection(signal_without_qrs, fs, w1_size, k_factor, r_peaks, ecg_signal_filtered_by25, d_max, d_min):
    _, p_peak_normal, _, _= t_wave_detection.t_peak_detection_aux(signal_without_qrs, fs, w1_size, k_factor, r_peaks, ecg_signal_filtered_by25, d_max, d_min)
    _, p_peak_low, _ , _= t_wave_detection.t_peak_detection_aux(-signal_without_qrs, fs, w1_size, k_factor, r_peaks, -ecg_signal_filtered_by25, d_max, d_min)
    p_united = np.sort(np.concatenate((p_peak_normal, p_peak_low)))
    return p_united , p_peak_normal, p_peak_low

def p_peaks_annotations(ecg_original, chosen_ann, seg=0, all_seg=False):
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
    p_annotations = np.zeros(len_ann, dtype=int)
    for index, marker in enumerate(annotations_markers):
        if marker == 'p':# and index != 0 and annotations_markers[index - 1] == '(':  # p_peak marker is 'p'
            p_annotations = np.insert(p_annotations, 0, annotations_samples[index])

    p_annotations = p_annotations[p_annotations != 0]
    p_annotations = np.sort(p_annotations)
    p_annotations = p_annotations - seg * signal_len_in_time * fs
    return p_annotations


def comparison_p_peaks(p_peaks_real_annotations, p_peaks_our_annotations, fs, r_intervals_size, margin_mistake_in_sec=0.030):
    distance_from_real = np.zeros(p_peaks_real_annotations.size, dtype=int)
    success = 0
    margin_mistake = round(margin_mistake_in_sec*fs)

    for i in range(distance_from_real.size):
        for j in range(p_peaks_our_annotations.size):
            if p_peaks_our_annotations[j] != -1:
                distance = abs(p_peaks_real_annotations[i] - p_peaks_our_annotations[j])
            else:
                distance = margin_mistake + 1
            if distance <= margin_mistake:
                distance_from_real[i] = 1
                p_peaks_our_annotations[j] = -1
                success = success + 1
                break  # break the t_peaks_our_ann for
        # if distance_from_real[i] == 0:
        #    print(p_peaks_real_annotations[i] / fs)

    return success, distance_from_real.size

