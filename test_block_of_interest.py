import numpy as np
import lead_extractor as le
# import plot_manager as pm
import qrs_detection as qrs
import csv
import t_wave_detection
import processing_functions as pf


# check signals 121 124 .


#with open('success rate t peaks per record per scale .csv', 'w', encoding='UTF8') as f:
success = 0
number_of_t_dots = 0
count = 0
w1_size = 0.070
for i in range(1, 201, 1):
    ecg_original = le.ecg_lead_ext('ludb', i, 'ii')
    fs = ecg_original['fs']
    ecg_original_copy = ecg_original.copy()
    ecg_processed = pf.ecg_pre_processing(ecg_original.copy())
    ecg_processed = qrs.detect_qrs(ecg_processed)
    ecg_original_copy['original_signal'] = pf.band_pass_filter(0.5, 12, ecg_original_copy['original_signal'], fs)
    q_ann, s_ann = qrs.find_q_s_ann(ecg_processed, True, True, realLabels=True)
    if q_ann.size <= 1 or s_ann.size <= 1:
        print(f'there is {q_ann.size} q annotations and {s_ann.size} s annotations in record {i}')
        continue
    signal_without_qrs = t_wave_detection.qrs_removal(ecg_original_copy, q_ann, s_ann)
    ecg_original_copy['original_signal'] = signal_without_qrs
    r_peaks = qrs.r_peaks_annotations(ecg_processed, 'real')
    if r_peaks.size <= 1:
        print(f'there is only {r_peaks.size} peaks')
        continue
    k_factor = 1
    first_boi, ma_peak, ma_t_wave = t_wave_detection.block_of_interest(signal_without_qrs, fs, w1_size, w1_size*2 , k_factor)
    # pm.plot_3_signals(ma_peak, ma_t_wave,first_boi,  fs, 'ma peak', 'ma t wave','initial block of interests')
    # pm.plot_2_signals(signal_without_qrs, first_boi, fs, 'signal without qrs', 'initial block of interests')
    ecg_signal_filtered = pf.band_pass_filter(0.5, 49, (ecg_original.copy())['original_signal'], fs)
    new_boi = t_wave_detection.thresholding(ecg_original_copy, first_boi, r_peaks, w1_size, k_factor)
    t_start, t_peak, t_end = t_wave_detection.find_real_blocks(ecg_original_copy, ecg_signal_filtered, r_peaks, new_boi)
    while t_peak.size != r_peaks.size - 1:  # think about localize on one interval RR
        k_factor = 0.9 * k_factor
        new_boi = t_wave_detection.thresholding(ecg_original_copy, first_boi, r_peaks, w1_size, k_factor)
        t_start, t_peak, t_end = t_wave_detection.find_real_blocks(ecg_original_copy, ecg_signal_filtered, r_peaks, new_boi)
        # pm.plot_2_signals(signal_without_qrs, new_boi, fs, 'signal without qrs', 'proceed block of interests')

    t_real_peaks = t_wave_detection.t_peaks_annotations(ecg_original_copy, 'real')
    success_record, number_of_t_dots_record = t_wave_detection.comparison_t_peaks(t_real_peaks.copy(), t_peak.copy(), fs)
    # TODO read again comparison function to understand whats going on there
    success += success_record
    number_of_t_dots += number_of_t_dots_record
        # print(success_record , number_of_t_dots_record)

    #if np.round(100 * success_record / number_of_t_dots_record) != 100:
        #pm.plot_signal_with_dots2(ecg_original['original_signal'], t_real_peaks, t_peak, fs, 'original signal', 'real t peaks', 'our t peaks', i)
        #count += 1
        #print(i, f'{success_record}/{number_of_t_dots_record}')

        #pm.plot_signal_with_dots(ecg_original['original_signal'], t_peak, fs, 'original signal', 't peaks', i)

    # writer = csv.writer(f)
        # write the data
    # writer.writerow(('ludb', i, 100 * np.round(success_record / number_of_t_dots_record, 3)))

    # writer = csv.writer(f)
    # writer.writerow(f'total success rate of ludb is {100 * np.round(success / number_of_t_dots, 4)}')
#print(100 * np.round(success / number_of_t_dots, 3), f'{success}/{number_of_t_dots}', count)
