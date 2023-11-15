import lead_extractor as le
import plot_manager as pm
import copy
import numpy as np
import scipy
import p_wave_detection
import t_wave_detection
import qrs_detection as qrs

dataset = 'qt'
w1_size = 0.030
success = 0
number_of_p_dots = 0
for i in range(28, 30, 1):
    signal_len_in_time = 700
    ecg_dict_original = le.ecg_lead_ext(signal_len_in_time, dataset, i, 'ii')
    ecg_dict_manually_ann = le.ecg_lead_ext(signal_len_in_time, dataset, i, 'ii', 'q1c')
    fs = ecg_dict_original['fs']
    ecg_dict_copy = copy.deepcopy(ecg_dict_original)
    #print(i)
    for seg in range(0,1):#ecg_dict_copy['num_of_segments']):
        signal = ecg_dict_copy['original_signal'][seg]
        b, a = scipy.signal.butter(2, [0.5, 12], btype='bandpass', output='ba', fs=fs)
        signal = scipy.signal.filtfilt(b, a, signal)
        p_real_peaks = p_wave_detection.p_peaks_annotations(ecg_dict_manually_ann, 'real', seg)
        r_peaks = qrs.r_peaks_annotations(ecg_dict_original, 'real', seg)
        #all_ann = np.array(ecg_dict_copy['ann'][seg])
        b2, a2 = scipy.signal.butter(2, 0.67, btype='highpass', output='ba', fs=fs)
        signal_without_dc = scipy.signal.filtfilt(b2, a2, ecg_dict_copy['original_signal'][seg])
        signal_without_dc = ecg_dict_original['original_signal'][seg]


        #pm.plot_signal_with_dots(ecg_dict_copy['original_signal'][seg], all_ann, fs)
        q_ann, s_ann = qrs.find_q_s_ann(ecg_dict_original, seg, True, True, realLabels=True)
        pvc_beats = p_wave_detection.pvc_detection(signal_without_dc, r_peaks, q_ann, s_ann, fs)
        if q_ann.size == 0 or s_ann.size == 0:
            continue
        signal_without_qrs = t_wave_detection.qrs_removal(signal, seg, q_ann, s_ann)

        p_peaks_united = p_wave_detection.p_peak_detection(signal_without_qrs, fs, w1_size, 1, r_peaks, signal_without_qrs, 0.999, 0.7)
        p_wave_detection.atrial_fib_detection(r_peaks, signal, p_peaks_united)
        #pm.plot_signal_with_dots2(signal_without_dc, p_real_peaks, p_peaks_united, fs, 'original signal', 'p_real_peaks', 'our p peaks', i, seg, signal_len_in_time)
        success_record, number_of_p_dots_record = p_wave_detection.comparison_p_peaks(p_real_peaks.copy(),
                                                                                      p_peaks_united.copy(), fs,
                                                                                      r_peaks.size - 1, 0.050)
        success += success_record
        number_of_p_dots += number_of_p_dots_record

        if number_of_p_dots_record != 0:
            score = np.round(100 * success_record / number_of_p_dots_record, 5)
            if score < 100:
                print(i, f'{success_record}/{number_of_p_dots_record}')
        pm.plot_signal_with_dots2(signal_without_dc, p_real_peaks, p_peaks_united, fs, 'original signal', 'p_real_peaks', 'our p peaks', i, seg, signal_len_in_time)
if number_of_p_dots != 0:
    print(round(success * 100 / number_of_p_dots, 5), f'{success}/{number_of_p_dots}')
