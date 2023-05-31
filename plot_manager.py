import matplotlib.pyplot as plt


def plot_single_signal(ecg_dict):
    # Calculate time array
    time = [i / ecg_dict['fs'] for i in range(len(ecg_dict['signal']))]

    # Plot the signal
    plt.plot(time, ecg_dict['signal'])
    plt.title(f'ECG Lead {ecg_dict["lead"]} for datafile {ecg_dict["name"]}')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (mV)')
    if ecg_dict['ann'] is not None:
        plt.scatter([time[i] for i in ecg_dict['ann']], [ecg_dict['signal'][i] for i in ecg_dict['ann']], c='r')
    plt.show()


def plot_original_vs_processed(signal1, signal2, ann=False):
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    time = [i / signal1['fs'] for i in range(len(signal1['signal']))]

    axs[0].plot(time, signal1['signal'])
    axs[0].set_ylabel('Amplitude (mV)')
    axs[0].set_title('Original ECG Signal')
    if ann:
        axs[0].scatter([time[i] for i in signal1['ann']], [signal1['signal'][i] for i in signal1['ann']], c='r')

    axs[1].plot(time, signal2['signal'])
    axs[1].set_ylabel('Amplitude (mV)')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_title('Processed ECG Signal')
    if ann:
        axs[1].scatter([time[i] for i in signal2['ann']], [signal2['signal'][i] for i in signal2['ann']], c='r')

    plt.show()
