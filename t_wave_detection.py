import numpy as np


def block_of_interest(y, fs, w1=70, w2=140):
    w1 *= fs
    w2 *= fs
    ma_peak = moving_average(y, w1)
    ma_t_wave = moving_average(y, w2)
    boi = []
    n = len(y)
    for i in range(n):
        boi.append(int(ma_peak > ma_t_wave))
    return boi


def moving_average(arr, w_size):
    # Initialize an empty list to store moving averages
    moving_averages = []
    n = len(arr)

    for i in range(n):
        min_index = max(0, round(i-n/2))
        max_index = min(n, round(i+n/2))
        # Calculate the average of current window
        window_average = round(np.sum(arr[min_index:max_index]) / w_size, 3)
        moving_averages.append(window_average)
    return moving_averages


if __name__ == "__main__":
    a = np.random.randint(10, size=20)
    print(a)
    print(moving_average(a, 4))
