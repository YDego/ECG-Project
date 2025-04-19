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
SIGNAL_LEN_FOR_MIT = 1806
WINDOW_SIZE_FOR_T_PEAKS = 0.070
WINDOW_SIZE_FOR_P_PEAKS = 0.030
dict_success = {}


def comparison_all_peaks(real_r_peaks, our_r_peaks, real_t_peaks, our_t_peaks, real_p_peaks, our_p_peaks, fs, signal_number):
    success_r_peaks, len_r_peaks = comparison_peaks(real_r_peaks.copy(), our_r_peaks.copy(), fs, 0.050)
    #print((len(real_r_peaks), len(our_r_peaks)))
    success_t_peaks, len_t_peaks = comparison_peaks(real_t_peaks.copy(), our_t_peaks.copy(), fs, 0.050)
    success_p_peaks, len_p_peaks = comparison_peaks(real_p_peaks.copy(), our_p_peaks.copy(), fs, 0.050)
    if len_p_peaks == 0 and len_t_peaks == 0:
        dict_success[str(signal_number)] = {'r': 100 * success_r_peaks / len_r_peaks, 't': None, 'p': None}
    elif len_p_peaks == 0:
        dict_success[str(signal_number)] = {'r': 100 * success_r_peaks / len_r_peaks, 't': 100 * success_t_peaks / len_t_peaks, 'p': None}
    else:
        dict_success[str(signal_number)] = {'r': 100*success_r_peaks/len_r_peaks, 't': 100*success_t_peaks/len_t_peaks, 'p': 100*success_p_peaks/len_p_peaks}
    return success_r_peaks, len_r_peaks, success_t_peaks, len_t_peaks, success_p_peaks, len_p_peaks


def comparison_peaks(peaks_real_annotations, peaks_our_annotations, fs, margin_mistake_in_sec=0.030):
    success = 0
    tolerance = round(margin_mistake_in_sec*fs)
    for index in range(peaks_real_annotations.size):
        for j in range(peaks_our_annotations.size):
            if peaks_our_annotations[j] != -1:
                distance = abs(peaks_real_annotations[index] - peaks_our_annotations[j])  # calc distance between real peak and our peak
            else:
                continue  # will not enter second if statement -> meaning we already found real peak that connect to this peak
            if distance <= tolerance:
                peaks_our_annotations[j] = -1  # set -1 because suitable peak was found to real peak
                success = success + 1
                break  # break the t_peaks_our_ann for
    return success, peaks_real_annotations.size


def main():
    dataset = ''
    num_sets = 0
    signal_len_in_time = 0
    all_signals = input("Perform QRS detection for all signals in ludb/qt/mit [input: l/q/m]? ")
    if all_signals == 'l' or all_signals == 'q' or all_signals == 'm':
        if all_signals == 'l':
            dataset = 'ludb'
            num_sets = 201
            signal_len_in_time = SIGNAL_LEN_FOR_LUDB
        elif all_signals == 'q':
            dataset = 'qt'
            num_sets = 106
            signal_len_in_time = SIGNAL_LEN_FOR_QT
        elif all_signals == 'm':
            dataset = 'mit'
            num_sets = 46
            signal_len_in_time = SIGNAL_LEN_FOR_MIT

        dict_all_success_peaks = {'all_r_success': 0,
                                  'len_all_real_r_peaks': 0,
                                  'all_t_success': 0,
                                  'len_all_real_t_peaks': 0,
                                  'all_p_success': 0,
                                  'len_all_real_p_peaks': 0}
        for i in range(1, num_sets, 1):
            if all_signals == 'm':
                ecg_original = le.ecg_lead_ext(signal_len_in_time, dataset, i)
            else:
                ecg_original = le.ecg_lead_ext(signal_len_in_time, dataset, i, 'ii')
            print(i)
            ecg_processed = pf.ecg_pre_processing(ecg_original)
            ecg_processed = qrs.detect_qrs(ecg_processed)
            ecg_original['our_ann'] = copy.deepcopy(ecg_processed['our_ann'])
            ecg_original['our_ann_markers'] = copy.deepcopy(ecg_processed['our_ann_markers'])
            our_q_peaks, our_s_peaks = qrs.find_q_s_ann(ecg_original, findQann=True, findSann=True, realLabels=False, all_seg=True)
            our_r_peaks = qrs.r_peaks_annotations(ecg_processed, 'our', all_seg=True)
            #print(len(our_r_peaks))
            our_t_peaks = t_wave_detection.main_t_peak_detection(ecg_original, WINDOW_SIZE_FOR_T_PEAKS, SIGNAL_LEN_FOR_LUDB, which_r_ann='our', real_q_s_ann=False)
            our_p_peaks = p_wave_detection.main_p_peak_detection(ecg_original, WINDOW_SIZE_FOR_P_PEAKS, SIGNAL_LEN_FOR_LUDB, which_r_ann='our', real_q_s_ann=False)
            real_r_peaks = qrs.r_peaks_annotations(ecg_processed, 'real', all_seg=True)
            real_t_peaks = t_wave_detection.t_peaks_annotations(ecg_original, 'real', all_seg=True)
            real_p_peaks = p_wave_detection.p_peaks_annotations(ecg_original, 'real', all_seg=True)
            all_united = np.sort(np.concatenate((our_p_peaks, our_t_peaks, our_r_peaks, our_q_peaks, our_s_peaks)))
            all_success_tuple = comparison_all_peaks(real_r_peaks, our_r_peaks, real_t_peaks, our_t_peaks, real_p_peaks, our_p_peaks, ecg_original['fs'], i)
            index = 0
            for keys in dict_all_success_peaks.keys():
                dict_all_success_peaks[keys] += all_success_tuple[index]
                index += 1
            #  np.array(ecg_original['ann'][0]) , 'all real annotations',
            #pm.plot_signal_with_dots(ecg_original['original_signal'][0], our_r_peaks, ecg_original['fs'], 'ecg signal', 'all our annotations',  i)

        for key, value in dict_success.items():
            print(key, ":", value)
        if dict_all_success_peaks["len_all_real_r_peaks"] != 0:
            print(f'score for all signals -->  r peaks: {round(100*(dict_all_success_peaks["all_r_success"] / dict_all_success_peaks["len_all_real_r_peaks"]), 3)}')
        if dict_all_success_peaks["len_all_real_t_peaks"] != 0:
            print(f'------------------------>  t peaks: {round(100*(dict_all_success_peaks["all_t_success"] / dict_all_success_peaks["len_all_real_t_peaks"]), 3)}')
        if dict_all_success_peaks["len_all_real_p_peaks"] != 0:
            print(f'------------------------>  p peaks: {round(100*(dict_all_success_peaks["all_p_success"] / dict_all_success_peaks["len_all_real_p_peaks"]),3)}')

        #print((success_final/number_of_dots_final)*100)
        #dict_bad_examples = {item: value for (item, value) in dict_success.items() if value < 90}
        #print(dict_bad_examples)
        #print(len(dict_bad_examples))

        # with open('score2.csv', 'w', encoding='UTF8') as f:
        #     writer = csv.writer(f)
        #
        #     # write the data
        #     writer.writerow(dict_success.items())
        #     writer.writerow(dict_bad_examples.items())

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



if __name__ == "__main__":
    main()