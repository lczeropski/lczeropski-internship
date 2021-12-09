#%%
import pandas as pd
import numpy as np
# %%
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
wine = pd.read_csv(url)
wine.head()
# %%
wine = wine.drop(wine.columns[[0,3,6,8,10,12,13]], axis = 1)
# %%
wine.head()
wine.columns = ['alcohol', 'malic_acid', 'alcalinity_of_ash', 'magnesium', 'flavanoids', 'proanthocyanins', 'hue']
# %%
wine.head()
# %%
wine.iloc[0:3,0] = np.nan
wine.head()
# %%
wine.iloc[2:4,3] = np.nan
wine.head()
# %%
wine.alcohol.fillna(10, inplace = True)
wine.magnesium.fillna(100, inplace = True)
wine.head()
# %%
wine.isnull().sum()
# %%
arr = np.random.randint(10, size = 10)
arr
# %%
wine.alcohol[arr] = np.nan
wine.head(10)
# %%
wine.isnull().sum()
# %%
wine = wine.dropna(axis = 0, how = "any")
#%%
wine.head()
# %%
nnull = wine.alcohol.notnull()
nnull
# %%
wine.alcohol[nnull]
#%%
wine = wine.reset_index(drop = True)
wine.head()