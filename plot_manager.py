import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines

marker_dict = {
    '(': '<',
    ')': '>',
    'p': 'X',
    'N': 'X',
    'n': 'o',
    't': 'X'
}

color_dict = {
    '(': 'k',
    ')': 'k',
    'p': 'b',
    'N': 'g',
    'n': 'g',
    't': 'r'
}

marker_title_dict = {
    '(': 'Opening point',
    ')': 'Closing point',
    'p': 'P peak',
    'N': 'R peak',
    'n': 'R peak',
    't': 't peak'
}

legends = []
for marker in marker_dict:
    legends.append(mlines.Line2D([], [], color=color_dict[marker], marker=marker_dict[marker], markersize=5, label=marker_title_dict[marker]))


def marker_converter(ann_markers):
    markers_converted = ann_markers[:]
    for i in range(len(ann_markers)):
        markers_converted[i] = marker_dict[ann_markers[i]]

    return markers_converted


def color_converter(ann_markers):
    colors = ann_markers[:]
    for i in range(len(ann_markers)):
        colors[i] = color_dict[ann_markers[i]]

    return colors


def plot_ann(ann, ann_markers, signal, time, plotter):
    markers = marker_converter(ann_markers.copy())
    colors = color_converter(ann_markers)
    j = 0

    for i in ann:
        plotter.scatter(time[i], signal[i], c=colors[j], marker=markers[j])
        j += 1


def plot_single_signal(ecg_dict, ann=False, our_ann=False):
    for seg in range(ecg_dict['num_of_segments']):
        fft = ecg_dict["fft"][seg]
        frequency_bins = ecg_dict["frequency_bins"][seg]
        signal = ecg_dict['original_signal'][seg]
        # Calculate time array
        time = [i / ecg_dict['fs'] for i in range(len(ecg_dict['original_signal'][seg]))]  # 16.6 change to ecg_dict['original_signal'] from ecg_dict['signal']

        # Plot the signal
        plt.subplot(2, 1, 1)
        plt.plot(time, signal)
        plt.title(f'Database: {ecg_dict["dataset"]}, datafile: {ecg_dict["name"]}, Lead {ecg_dict["lead"]}')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (mV)')
        if ann:
            plot_ann(ecg_dict['ann'][seg], ecg_dict['ann_markers'][seg], signal, time, plt)

        if our_ann:
            plot_ann(ecg_dict['our_ann'][seg], ecg_dict['our_ann_markers'][seg], signal, time, plt)

        # Plot the FFT
        plt.subplot(2, 1, 2)
        plt.plot(frequency_bins, np.abs(fft), color='red')
        plt.xlabel('Frequency')
        plt.ylabel('Magnitude')
        plt.title('FFT')

        # Display the plot
        plt.tight_layout()
        plt.show()


def plot_original_vs_processed(ecg_dict_1, ecg_dict_2, ann=False, our_ann=False):
    signal1 = ecg_dict_1["signal"]
    signal2 = ecg_dict_2["signal"]
    fft1 = ecg_dict_1["fft"]
    fft2 = ecg_dict_2["fft"]
    freq_bin1 = ecg_dict_1["frequency_bins"]
    freq_bin2 = ecg_dict_2["frequency_bins"]

    fig, axs = plt.subplots(2, 2)
    fs = ecg_dict_1['fs']
    time = [i / fs for i in range(len(signal1))]

    # Plot the signal
    axs[0, 0].plot(time, signal1)
    axs[0, 0].set_ylabel('Amplitude (mV)')
    axs[0, 0].set_title('Original ECG Signal')
    if ann:
        plot_ann(ecg_dict_1['ann'], ecg_dict_1['ann_markers'], signal1, time, axs[0, 0])
        axs[0, 0].legend(handles=legends, fontsize="7",  loc="upper left")

    # Plot the FFT
    axs[1, 0].plot(freq_bin1, np.abs(fft1))
    axs[1, 0].set_xlabel('Frequency')
    axs[1, 0].set_ylabel('Magnitude')
    axs[1, 0].set_title('FFT')

    # Plot the signal
    axs[0, 1].plot(time, signal2, color='red')
    axs[0, 1].set_ylabel('Amplitude (mV)')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_title('Processed ECG Signal')
    if ann or our_ann:
        axs[0, 1].legend(handles=legends, fontsize="7",  loc="upper left")
        if ann:
            plot_ann(ecg_dict_2['ann'], ecg_dict_2['ann_markers'], signal2, time, axs[0, 1])
        if our_ann:
            plot_ann(ecg_dict_2['our_ann'], ecg_dict_2['our_ann_markers'], signal2, time, axs[0, 1])

    # Plot the FFT
    axs[1, 1].plot(freq_bin2, np.abs(fft2), color='red')
    axs[1, 1].set_xlabel('Frequency')
    axs[1, 1].set_ylabel('Magnitude')
    axs[1, 1].set_title('FFT')

    axs[0, 0].sharex(axs[0, 1])
    axs[0, 0].sharey(axs[0, 1])
    axs[1, 0].sharex(axs[1, 1])
    axs[1, 0].sharey(axs[1, 1])

    plt.show()
