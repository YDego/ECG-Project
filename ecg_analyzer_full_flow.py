import processing_functions as pf
import lead_extractor as le
import plot_manager as pm
import csv

all_signals = input("Perform QRS detection for all signals in ludb [y/N]? ")

if all_signals == 'y':
    success_final = 0
    number_of_dots_final = 0
    dict_success = {}
    for i in range(1,201,1):
        ecg_original = le.ecg_lead_ext('ludb', i, 'ii')
        ecg_processed = pf.ecg_pre_processing(ecg_original)
        success_final = success_final + ecg_processed["r_peak_success"][0]
        number_of_dots_final = number_of_dots_final + ecg_processed["r_peak_success"][1]
        dict_success[str(i)] = 100*(ecg_processed["r_peak_success"][0]/ecg_processed["r_peak_success"][1])
    print(dict_success)
    print((success_final/number_of_dots_final)*100)
    dict_bad_examples = {item: value for (item, value) in dict_success.items() if value != 100}
    print(dict_bad_examples)
    print(len(dict_bad_examples))




    with open('score.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the data
        writer.writerow(dict_success.items())
        writer.writerow(dict_bad_examples.items())
else:

    # Call the function with leads and file_count as inputs

    ecg_original = le.ecg_lead_ext()

    # Plot signal
    pm.plot_single_signal(ecg_original)

    # ECG processing

    ecg_processed = pf.ecg_pre_processing(ecg_original)
    pm.plot_single_signal(ecg_processed)

    # Plot original vs. processed signal
    pm.plot_original_vs_processed(ecg_original, ecg_processed)