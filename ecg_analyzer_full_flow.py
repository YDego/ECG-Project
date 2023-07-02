import processing_functions as pf
import lead_extractor as le
import plot_manager as pm
import csv
import qrs_detection as qrs

all_signals = input("Perform QRS detection for all signals in ludb/qt/mit [l/q/m]? ")

if all_signals == 'l' or all_signals == 'q' or all_signals == 'm':
    success_final = 0
    number_of_dots_final = 0
    dict_success = {}
    if all_signals == 'l':
        for i in range(1, 201, 1):
            ecg_original = le.ecg_lead_ext('ludb', i, 'ii')
            ecg_processed = pf.ecg_pre_processing(ecg_original)
            ecg_processed = qrs.detect_qrs(ecg_processed)
            ecg_processed = qrs.comparison_r_peaks(ecg_processed)
            success_final = success_final + ecg_processed["r_peak_success"][0]
            number_of_dots_final = number_of_dots_final + ecg_processed["r_peak_success"][1]
            dict_success[str(i)] = 100*(ecg_processed["r_peak_success"][0]/ecg_processed["r_peak_success"][1])
    elif all_signals == 'q':
        for i in range(1, 106, 1):
            ecg_original = le.ecg_lead_ext('qt', i, 'ii')
            ecg_processed = pf.ecg_pre_processing(ecg_original)
            ecg_processed = qrs.detect_qrs(ecg_processed)
            ecg_processed = qrs.comparison_r_peaks(ecg_processed)
            success_final = success_final + ecg_processed["r_peak_success"][0]
            number_of_dots_final = number_of_dots_final + ecg_processed["r_peak_success"][1]
            if ecg_processed["r_peak_success"][1] != 0:
                dict_success[str(i)] = 100 * (ecg_processed["r_peak_success"][0] / ecg_processed["r_peak_success"][1])
            else:
                dict_success[str(i)] = 0
    else:
        for i in range(1, 46, 1):
            ecg_original = le.ecg_lead_ext('mit', i)
            ecg_processed = pf.ecg_pre_processing(ecg_original)
            ecg_processed = qrs.detect_qrs(ecg_processed)
            ecg_processed = qrs.comparison_r_peaks(ecg_processed)
            # pm.plot_single_signal(ecg_processed)
            success_final = success_final + ecg_processed["r_peak_success"][0]
            number_of_dots_final = number_of_dots_final + ecg_processed["r_peak_success"][1]
            # if 100 * (ecg_processed["r_peak_success"][0] / ecg_processed["r_peak_success"][1]) != 100:
            #     pm.plot_single_signal(ecg_processed)
            #     print(i)
            if ecg_processed["r_peak_success"][1] != 0:
                dict_success[str(i)] = 100 * (ecg_processed["r_peak_success"][0] / ecg_processed["r_peak_success"][1])
            else:
                dict_success[str(i)] = 0

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
    # Call the function with leads and file_count as inputs
    ecg_original = le.ecg_lead_ext()
    pm.plot_single_signal(ecg_original)

    # ECG pre-processing
    ecg_processed = pf.ecg_pre_processing(ecg_original)
    pm.plot_original_vs_processed(ecg_original, ecg_processed)

    # QRS Detection
    ecg_qrs = qrs.detect_qrs(ecg_processed)
    ecg_qrs = qrs.comparison_r_peaks(ecg_qrs)

    ecg_processed["ann"] = qrs.r_peaks_annotations(ecg_processed, 'real')
    ecg_processed["signal"] = ecg_original["signal"]
    pm.plot_original_vs_processed(ecg_original, ecg_processed, True, True)
