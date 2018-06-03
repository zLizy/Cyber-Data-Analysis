import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.robust import mad
import pywt

def waveletDenoising(x, wavelet="db4", level=1):
    # calculate the wavelet coefficients
    coeff = pywt.wavedec(x, wavelet,mode='symmetric')
    # calculate a threshold
    sigma = mad(coeff[-level])

    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft")
                 for i in coeff[1:])
    # reconstruct the signal using the thresholded coefficients
    y = pywt.waverec(coeff, wavelet, mode="symmetric")
    return y[0:-1]
