import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import lead_extractor as le
import processing_functions as pr
from tkinter import ttk
import numpy as np
import plot_manager as pm


# Function to handle dataset selection
def select_dataset(event):
    selected_dataset = dataset_combobox.get()
    update_data_numbers(selected_dataset)


# Function to update the data number dropdown based on the selected dataset
def update_data_numbers(dataset_name):

    dataset = le.get_dataset(dataset_name)
    _, num_records = le.get_records(dataset)
    data_numbers = [i+1 for i in range(num_records)]
    data_number_combobox['values'] = data_numbers
    data_number_combobox.configure(state=tk.WRITABLE)
    update_leads(dataset)


# Function to update the leads dropdown
def update_leads(dataset):
    leads_combobox['values'] = dataset['leads']
    leads_combobox.configure(state=tk.WRITABLE)
    plot_button.configure(state=tk.NORMAL)


# Function to handle plot button click
def plot_ecg():
    selected_dataset = dataset_combobox.get()
    selected_data_number = data_number_combobox.get()
    selected_lead = leads_combobox.get()
    ecg_dict = le.ecg_lead_ext(selected_dataset, int(selected_data_number), selected_lead)

    # Compute FFT
    ecg_dict["fft"], ecg_dict["frequency_bins"] = pr.compute_fft(ecg_dict["signal"], ecg_dict["fs"])

    # Update GUI with both plots
    plot_ecg_data(ecg_dict)


# Function to load ECG data based on the selected dataset and data number
def load_ecg_data(dataset, data_number):
    ecg_dict = le.ecg_lead_ext(dataset)  # Calling the ecg_lead_ext function from lead_extractor module
    # Here you can perform any necessary processing on the ECG data based on the selected data_number

    ecg_processed_signal = pr.ecg_pre_processing(ecg_dict)

    return ecg_processed_signal


# Function to plot ECG data
def plot_ecg_data(ecg_dict):
    fft = ecg_dict["fft"]
    frequency_bins = ecg_dict["frequency_bins"]

    # Calculate time array
    time = [i / ecg_dict['fs'] for i in range(len(ecg_dict['signal']))]

    # Plot the signal
    plt.subplot(2, 1, 1)
    plt.plot(time, ecg_dict['signal'])
    plt.title(f'ECG Lead {ecg_dict["lead"]} for datafile {ecg_dict["name"]}')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (mV)')
    if ecg_dict['ann'] is not None:
        ann = ecg_dict['ann']
        markers = pm.convert_markers(ecg_dict['ann_markers'])
        colors = pm.color_coverter(ecg_dict['ann_markers'])
        j = 0

        for i in ann:
            plt.scatter(time[i], ecg_dict['signal'][i], c=colors[j], marker=markers[j])
            j += 1

    # Plot the FFT
    plt.subplot(2, 1, 2)
    plt.plot(frequency_bins, np.abs(fft), color='red')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.title('FFT')

    # Create a Tkinter canvas and display the plot
    canvas = FigureCanvasTkAgg(plt.gcf(), master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


root = tk.Tk()
root.title("ECG Data Plotter")
root.geometry("800x600")

# List of available ECG datasets
dataset_list = ["ludb", "qt"]  # Add more dataset names as needed

dataset_label = tk.Label(root, text="Choose an ECG dataset:")
dataset_label.pack()

dataset_combobox = tk.ttk.Combobox(root, values=dataset_list)
dataset_combobox.bind("<<ComboboxSelected>>", select_dataset)
dataset_combobox.pack()

data_number_label = tk.Label(root, text="Choose a data number:")
data_number_label.pack()

data_number_combobox = tk.ttk.Combobox(root, state=tk.DISABLED)
data_number_combobox.pack()

leads_label = tk.Label(root, text="Choose a lead:")
leads_label.pack()

leads_combobox = tk.ttk.Combobox(root, state=tk.DISABLED)
leads_combobox.pack()

plot_button = tk.Button(root, text="Plot ECG Data", command=plot_ecg, state=tk.DISABLED)
plot_button.pack()

root.mainloop()
