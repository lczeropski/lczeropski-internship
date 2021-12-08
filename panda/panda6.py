#%%
import pandas as pd
# %%
baby_names = pd.read_csv('https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/06_Stats/US_Baby_Names/US_Baby_Names_right.csv')
baby_names
# %%
baby_names.head(10)
# %%
del baby_names['Unnamed: 0']
# %%
del baby_names['Id']
# %%
baby_names.head()
# %%
baby_names['Gender'].value_counts()
# %%
# %%
del baby_names["Year"]
#%%
names = baby_names.groupby("Name").sum()
# %%
names = names.sort_values("Count", ascending = 0)
# %%
len(names)
# %%
names.head(1) #Jacob
# %%
len(names[names.Count == names.Count.min()])
# %%
names[names.Count == names.Count.median()]
# %%
names.Count.std()
# %%
names.describe()
# %%
