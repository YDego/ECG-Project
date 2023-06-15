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



##########  shahar
# new_signal, freq, spectrum = filters_by_fft.band_pass_filter(8, 50, signal, record.fs)
# plt.plot(original_signal)
# q_and_s_annotation = qrs_detection.annotation_for_q_and_s(annotation_sample)
# plt.scatter(q_and_s_annotation, original_signal[q_and_s_annotation], c='k')  #
# # plt.scatter(annotation_sample,original_signal[annotation_sample],c='k')
# threshold = np.mean(abs(new_signal) ** 3)
# open_dots, R_peaks, closed_dots, all_dots = qrs_detection.detection_qrs(original_signal, abs(new_signal) ** 3,
#                                                                         threshold)  #
# plt.scatter(R_peaks, original_signal[R_peaks], c='b')
# # plt.scatter(local_peaks,original_signal[local_peaks], c='k')
# distance_from_real = qrs_detection.distance_from_real_dot(q_and_s_annotation, all_dots)
