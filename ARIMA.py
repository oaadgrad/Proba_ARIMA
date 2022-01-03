import numpy as np
import pandas as pd
import itertools
from datetime import datetime
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA

# Считывание временного ряда
timeseries = pd.read_csv('_btcusdt_1DAY.csv')
print(timeseries)
df = pd.DataFrame(timeseries, columns=['Date', 'Close'])
print(df)
df['Date'] = df['Date'].transform(lambda x: datetime.strptime(x, '%Y-%m-%d'))
df.set_index(keys='Date', drop=True, inplace=True)
df = df.squeeze(axis=1)
print(df)

#ARIMA перебор параметров


import warnings
warnings.filterwarnings("ignore")

p = range(0, 100)
d = q = range(0, 20)
pdq = list(itertools.product(p, d, q))
best_pdq = (0,0,0)
best_aic = np.inf
for params in pdq:
  model_test = ARIMA(df, order = params)
  result_test = model_test.fit()
  if result_test.aic < best_aic:
      best_pdq = params
      best_aic = result_test.aic
print(best_pdq, best_aic)


# (p,d,q) = (9,2,1)

#model = ARIMA(df, order=(9, 2, 1)) #вставьте свои числа вместо p, d и q
#result = model.fit()
#result.summary()
#result.plot_diagnostics(figsize=(15, 10))
#plt.show()