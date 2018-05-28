import numpy as np
from saxpy.sax import sax_via_window
from pandas import read_csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

def parser(x):
    return pd.datetime.strptime('2015'+x, '%Y%d/%m/%H %M')

data = read_csv('./BATADAL_dataset03.csv', header=0, parse_dates=[0], index_col=0, squeeze=True,\
                  date_parser=parser)

series = data.copy()

