import processing_functions
import lead_extractor
import plot_manager
# import qrs_detection


# Call the function with leads and file_count as inputs
ecg_original = lead_extractor.ecg_lead_ext()

# Plot signal
ecg_original["fft"], ecg_original["frequency_bins"] = processing_functions.compute_fft(ecg_original["signal"], ecg_original["fs"])
plot_manager.plot_single_signal(ecg_original)

# ECG processing
ecg_processed = processing_functions.ecg_pre_processing(ecg_original)

# ecg_wavelet = ecg_original.copy()

# ecg_wavelet["signal"] = qrs_detection.wavelet_filter(ecg_original["signal"])

ecg_processed["fft"], ecg_processed["frequency_bins"] = processing_functions.compute_fft(ecg_processed["signal"], ecg_processed["fs"])

plot_manager.plot_original_vs_processed(ecg_original, ecg_processed)
