
import numpy as np
from scipy.signal import correlate as cr


def correlation_signals(signal1, signal2):
    len_signal1, len_signal2 = len(signal1), len(signal2)
    if len_signal1 > len_signal2:
        longer_signal = signal1
        shorter_signal = signal2
    else:
        longer_signal = signal2
        shorter_signal = signal1
    correlation_len = len(shorter_signal)
    correlation_list = []
    for num_of_corr in range(0, len(longer_signal) - correlation_len + 1, 1):
        sum_of_one_correlation = 0
        sum_of_longer_signal_energy = 0
        sum_of_shorter_signal_energy = 0
        for index in range(0, correlation_len, 1):
            sum_of_one_correlation = sum_of_one_correlation + longer_signal[index+num_of_corr]*shorter_signal[index]
            sum_of_longer_signal_energy = sum_of_longer_signal_energy + longer_signal[index+num_of_corr]*longer_signal[index+num_of_corr]
            sum_of_shorter_signal_energy = sum_of_shorter_signal_energy + shorter_signal[index]*shorter_signal[index]
        correlation_list.append(sum_of_one_correlation / (sum_of_shorter_signal_energy*sum_of_longer_signal_energy)**0.5)

    print(correlation_list)
    start_of_best_correlation = correlation_list.index(max(correlation_list))
    print(start_of_best_correlation)
    return longer_signal[start_of_best_correlation:start_of_best_correlation + correlation_len]









print(correlation_signals([1,2,3,9,10,2,-1,-2,-3,-4,-5,-2,0,9], [-1,-40,-3]))

