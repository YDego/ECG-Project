import processing_functions as pf
import lead_extractor as le
import plot_manager as pm
# import qrs_detection


# Call the function with leads and file_count as inputs
ecg_original = le.ecg_lead_ext()

# Plot signal
pm.plot_single_signal(ecg_original)

# ECG processing
ecg_processed = pf.ecg_pre_processing(ecg_original)

# Plot original vs. processed signal
pm.plot_original_vs_processed(ecg_original, ecg_processed)
