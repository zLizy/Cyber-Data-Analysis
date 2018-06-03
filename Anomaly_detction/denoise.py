import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('signal.txt', sep="\n", header=None)

signal = list(df.values.flatten())

plt.plot(signal)
plt.show()

fft_values = np.fft.fft(signal)

plt.plot(np.abs(fft_values))
plt.show()

mean_value = np.mean(np.abs(fft_values))
threshold = 1.1*mean_value

fft_values[np.abs(fft_values) < threshold] = 0

filtered_samples = np.fft.ifft(fft_values)

plt.plot(np.abs(filtered_samples))
plt.show()  
