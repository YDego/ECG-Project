import processing_functions
import qrs_detection
import lead_extractor
import plot_manager


while input("Choose new datafile? [Y/n]: ") != 'n':
    # Call the function with leads and file_count as inputs
    ecg_original = lead_extractor.choose_lead_from_dataset()

    # Plot signal
    plot_manager.plot_single_signal(ecg_original)

    while input("Keep processing? [Y/n]: ") != 'n':
        # ECG processing
        ecg_processed = processing_functions.ecg_pre_processing(ecg_original)

        # Plot
        plot_manager.plot_original_vs_processed(ecg_original, ecg_processed)
