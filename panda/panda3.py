#%% grouping
import pandas as pd

# %%
drinks = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/drinks.csv')
drinks
# %%
drinks.groupby('continent').beer_servings.mean().sort_values(ascending = False)
# %%
#describe
drinks.groupby('continent').wine_servings.describe()
# %%
drinks.groupby('continent').mean()
# %%
drinks.groupby('continent').median()
# %%
drinks.groupby('continent').spirit_servings.agg(['mean', 'min', 'max'])
# %%
