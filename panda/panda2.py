#%%
import pandas as pd
import numpy as np
food = pd.read_csv('/Users/lczeropski/Downloads/en.openfoodfacts.org.products.tsv', sep='\t')
# %%
food.head()
# %%
food.shape
# %%
food.info()
# %%
food.columns
# %%
food.columns[104]
# %%
food.dtypes['-glucose_100g']
# %%
food.index
# %%
food.values[18][7]
# %%
