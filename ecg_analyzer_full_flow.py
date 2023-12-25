import copy

import p_wave_detection
import processing_functions as pf
import lead_extractor as le
import plot_manager as pm
import csv
import qrs_detection as qrs
import t_wave_detection
import numpy as np

SIGNAL_LEN_FOR_LUDB = 10
SIGNAL_LEN_FOR_QT = 900
SIGNAL_LEN_FOR_MIT = 1800
WINDOW_SIZE_FOR_T_PEAKS = 0.070
WINDOW_SIZE_FOR_P_PEAKS = 0.030

all_signals = input("Perform QRS detection for all signals in ludb/qt/mit [input: l/q/m]? ")
if all_signals == 'l' or all_signals == 'q' or all_signals == 'm':
    success_final = 0
    number_of_dots_final = 1
    dict_success = {}
    if all_signals == 'l':
        signal_len_in_time = SIGNAL_LEN_FOR_LUDB
        for i in range(1, 201, 1):
            ecg_original = le.ecg_lead_ext(signal_len_in_time, 'ludb', i, 'ii')
            ecg_processed = pf.ecg_pre_processing(ecg_original)
            ecg_processed = qrs.detect_qrs(ecg_processed)
            ecg_original['our_ann'] = copy.deepcopy(ecg_processed['our_ann'])
            ecg_original['our_ann_markers'] = copy.deepcopy(ecg_processed['our_ann_markers'])
            our_t_peaks = t_wave_detection.main_t_peak_detection(ecg_original, WINDOW_SIZE_FOR_T_PEAKS, SIGNAL_LEN_FOR_LUDB, which_r_ann='our')
            our_p_peaks = p_wave_detection.main_p_peak_detection(ecg_original, WINDOW_SIZE_FOR_P_PEAKS, SIGNAL_LEN_FOR_LUDB, which_r_ann='our')
            our_r_peaks_list = []
            for seg in range(ecg_processed['num_of_segments']):
                our_r_peaks_list.extend(qrs.r_peaks_annotations(ecg_processed, 'our', seg))
            our_r_peaks = np.array(our_r_peaks_list)
            all_united = np.sort(np.concatenate((our_p_peaks, our_t_peaks, our_r_peaks))) ## todo , in our r peaks array this is array in array.
            pm.plot_signal_with_dots(ecg_original['original_signal'][0], all_united, ecg_original['fs'], 'ecg signal', 'all annotations', i)
            ecg_processed = qrs.comparison_r_peaks(ecg_processed)
            for seg in range(ecg_processed['num_of_segments']):
                success_final = success_final + ecg_processed["r_peak_success"][seg][0]
                number_of_dots_final = number_of_dots_final + ecg_processed["r_peak_success"][seg][1]
                dict_success[str(i)] = 100*(ecg_processed["r_peak_success"][seg][0]/ecg_processed["r_peak_success"][seg][1])
    elif all_signals == 'q':
        signal_len_in_time = SIGNAL_LEN_FOR_QT
        for i in range(1, 106, 1):
            ecg_original = le.ecg_lead_ext(signal_len_in_time, 'qt', i, 'ii')
            ecg_processed = pf.ecg_pre_processing(ecg_original)
            ecg_processed = qrs.detect_qrs(ecg_processed)
            ecg_original['our_ann'] = copy.deepcopy(ecg_processed['our_ann'])
            ecg_original['our_ann_markers'] = copy.deepcopy(ecg_processed['our_ann_markers'])
            our_t_peaks = t_wave_detection.main_t_peak_detection(ecg_original, WINDOW_SIZE_FOR_T_PEAKS,
                                                                 SIGNAL_LEN_FOR_LUDB, which_r_ann='our')
            our_p_peaks = p_wave_detection.main_p_peak_detection(ecg_original, WINDOW_SIZE_FOR_P_PEAKS,
                                                                 SIGNAL_LEN_FOR_LUDB, which_r_ann='our')
            our_r_peaks_list = []
            for seg in range(ecg_processed['num_of_segments']):
                our_r_peaks_list.extend(qrs.r_peaks_annotations(ecg_processed, 'our', seg))
            our_r_peaks = np.array(our_r_peaks_list)
            all_united = np.sort(np.concatenate(
                (our_p_peaks, our_t_peaks, our_r_peaks)))  ## todo , in our r peaks array this is array in array.
            pm.plot_signal_with_dots(ecg_original['original_signal'][0], all_united, ecg_original['fs'], 'ecg signal',
                                     'all annotations', i)
            #success_final = success_final + ecg_processed["r_peak_success"][0]
            #number_of_dots_final = number_of_dots_final + ecg_processed["r_peak_success"][1]
            #if ecg_processed["r_peak_success"][1] != 0:
            #    dict_success[str(i)] = 100 * (ecg_processed["r_peak_success"][0] / ecg_processed["r_peak_success"][1])
            #else:
            #    dict_success[str(i)] = 0
    else:
        for i in range(1, 46, 1):
            signal_len_in_time = SIGNAL_LEN_FOR_MIT
            ecg_original = le.ecg_lead_ext(signal_len_in_time, 'mit', i)
            ecg_processed = pf.ecg_pre_processing(ecg_original)
            ecg_processed = qrs.detect_qrs(ecg_processed)
            ecg_original['our_ann'] = copy.deepcopy(ecg_processed['our_ann'])
            ecg_original['our_ann_markers'] = copy.deepcopy(ecg_processed['our_ann_markers'])
            our_t_peaks = t_wave_detection.main_t_peak_detection(ecg_original, WINDOW_SIZE_FOR_T_PEAKS,
                                                                 SIGNAL_LEN_FOR_LUDB, which_r_ann='our')
            our_p_peaks = p_wave_detection.main_p_peak_detection(ecg_original, WINDOW_SIZE_FOR_P_PEAKS,
                                                                 SIGNAL_LEN_FOR_LUDB, which_r_ann='our')
            our_r_peaks_list = []
            for seg in range(ecg_processed['num_of_segments']):
                our_r_peaks_list.extend(qrs.r_peaks_annotations(ecg_processed, 'our', seg))
            our_r_peaks = np.array(our_r_peaks_list)
            all_united = np.sort(np.concatenate(
                (our_p_peaks, our_t_peaks, our_r_peaks)))  ## todo , in our r peaks array this is array in array.
            pm.plot_signal_with_dots(ecg_original['original_signal'][0], all_united, ecg_original['fs'], 'ecg signal',
                                     'all annotations', i)

            #success_final = success_final + ecg_processed["r_peak_success"][0]
            #number_of_dots_final = number_of_dots_final + ecg_processed["r_peak_success"][1]
            # if 100 * (ecg_processed["r_peak_success"][0] / ecg_processed["r_peak_success"][1]) != 100:
            #     pm.plot_single_signal(ecg_processed)
            #     print(i)
            #if ecg_processed["r_peak_success"][1] != 0:
            #    dict_success[str(i)] = 100 * (ecg_processed["r_peak_success"][0] / ecg_processed["r_peak_success"][1])
            #else:
            #    dict_success[str(i)] = 0
    print(dict_success)
    print((success_final/number_of_dots_final)*100)
    dict_bad_examples = {item: value for (item, value) in dict_success.items() if value < 90}
    print(dict_bad_examples)
    print(len(dict_bad_examples))

    with open('score2.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the data
        writer.writerow(dict_success.items())
        writer.writerow(dict_bad_examples.items())

else:

    # Parameters
    show_all_segments = False
    plot_ann = False
    plot_our_ann = False

    # Call the function with leads and file_count as inputs
    ecg_original = le.ecg_lead_ext()
    if not show_all_segments:
        seg = int(input('Choose a segment to plot from 0 to {} (default 0): '.format(ecg_original['num_of_segments'] - 1)) or 0)
    else:
        seg = 0
    pm.plot_single_signal(ecg_original, seg, show_all_segments)

    # ECG pre-processing
    ecg_processed = pf.ecg_pre_processing(ecg_original)
    pm.plot_original_vs_processed(ecg_original, ecg_processed, seg, show_all_segments, plot_ann, plot_our_ann)

    # QRS Detection
    ecg_qrs = qrs.detect_qrs(ecg_processed)
    ecg_qrs = qrs.comparison_r_peaks(ecg_qrs)

    ecg_processed["signal"] = ecg_original["signal"]
    pm.plot_original_vs_processed(ecg_original, ecg_processed, seg, show_all_segments, plot_ann, plot_our_ann)
