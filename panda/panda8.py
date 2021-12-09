#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %%
url = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/09_Time_Series/Apple_Stock/appl_1980_2014.csv'
apple = pd.read_csv(url)
#%%
apple.head()
# %%
apple.dtypes
# %%
apple.Date = pd.to_datetime(apple.Date)
# %%
apple.Date.head()
# %%
apple = apple.set_index('Date')
# %%
apple.head()
#%%
apple.index.is_unique
# %%
apple.sort_index(ascending = True).head()
# %%
apple_month = apple.resample('BM').mean()
# %%
apple_month.head()
# %%
diff = apple.index.max() - apple.index.min()
# %%
diff.days
# %%
len(apple_month)
# %%
to_plot = apple['Adj Close'].plot(title = 'Apple Stock')
fig = to_plot.get_figure()
fig.set_size_inches(13.5,9)
# %%
