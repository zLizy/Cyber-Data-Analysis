import ruptures as rpt
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

def parser(x):
    return pd.datetime.strptime('2015'+x, '%Y%d/%m/%H %M')

data = read_csv('./BATADAL_dataset03.csv', header=0, parse_dates=[0], index_col=0, squeeze=True,\
                  date_parser=parser)

#n_samples, dim, sigma = 1000, 3, 4
n_bkps = 170  # number of breakpoints
#signal, bkps = rpt.pw_constant(n_samples, dim, n_bkps, noise_std=sigma)
#print(bkps)

signal=np.array(data['L_T3'])[range(1000)]
# detection
model = "l2"  # "l1", "rbf", "linear", "normal", "ar"
#algo = rpt.Window(width=24, model=model).fit(signal)
algo = rpt.Dynp(model=model, min_size=3, jump=5).fit(signal)
#algo = rpt.Window(model="rbf").fit(signal)
result = algo.predict(n_bkps)


#result = algo.predict(pen=100)
print('***')
# display
rpt.display(signal, result)
#plt.plot(result)
print(result)
plt.show()