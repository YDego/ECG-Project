import numpy as np
import t_wave_detection


def pvc_detection(signal, r_peaks, q_peaks, s_peaks, fs):
    r_peaks_size = r_peaks.size
    pvc_beats = np.zeros(r_peaks_size, dtype=bool)
    auc_beats = np.zeros(r_peaks_size, dtype=float)  # auc = area under qrs curve
    for index in range(r_peaks_size):
        #signal_seg = signal[q_peaks[index]:s_peaks[index]]
        signal_seg = signal[(r_peaks[index] - int(0.050 * fs)):(r_peaks[index] + int(0.050 * fs))]
        # auc_beats[index] = np.sum(np.abs(signal_seg))
        auc_beats[index] = np.trapz(np.abs(signal_seg), dx=1/fs)
    median = np.median(auc_beats)
    #print(auc_beats)
    for index in range(r_peaks_size):
        if auc_beats[index] > 1.3 * median:
            pvc_beats[index] = True
    #        print(r_peaks[index] / fs)

    return pvc_beats


def atrial_fib_detection(r_peaks, signal, p_peaks):
    rr_intervals = np.diff(r_peaks)
    print(rr_intervals)
    median_interval = np.median(rr_intervals)
    print(median_interval)
    irregular_beats = False
    for index in range(0, rr_intervals.size - 2):  # sliding window of 3 intervals
        count = 0
        for slide in range(index, index + 3, 1):
            if rr_intervals[slide] > median_interval * 1.5:
                count += 1
                print('irregular 1')
        if count == 2:
            irregular_beats = True
            print('irregular_beats')

    return






def p_peak_detection(signal_without_qrs, fs, w1_size, k_factor, r_peaks, ecg_signal_filtered_by25, d_max, d_min):
    _, p_peak_normal, _, _= t_wave_detection.t_peak_detection(signal_without_qrs, fs, w1_size, k_factor, r_peaks, ecg_signal_filtered_by25, d_max, d_min)
    _, p_peak_low, _ , _= t_wave_detection.t_peak_detection(-signal_without_qrs, fs, w1_size, k_factor, r_peaks, -ecg_signal_filtered_by25, d_max, d_min)
    p_united = np.sort(np.concatenate((p_peak_normal, p_peak_low)))
    return p_united
def p_peaks_annotations(ecg_original, chosen_ann, seg):
    fs = ecg_original['fs']
    signal_len_in_time = ecg_original['signal_len']
    if chosen_ann == "real":
        annotations_samples = ecg_original["ann"][seg]
        annotations_markers = ecg_original["ann_markers"][seg]
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
        #if distance_from_real[i] == 0:
        #    print(p_peaks_real_annotations[i] / fs)

    return success, distance_from_real.size

