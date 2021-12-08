#%%
import pandas as pd
import numpy as np
# %%
csv_url = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/04_Apply/Students_Alcohol_Consumption/student-mat.csv'
df = pd.read_csv(csv_url)
df.head()
# %%
sch_to_guar = df.loc[: , "school":"guardian"]
sch_to_guar
# %%
cap = lambda x: x.capitalize()
# %%
sch_to_guar['Mjob']=sch_to_guar['Mjob'].apply(cap)
sch_to_guar['Fjob']=sch_to_guar['Fjob'].apply(cap)
# %%
sch_to_guar.tail()
# %%
def majority(x):
    if x > 17:
        return True
    return False
sch_to_guar['legal_drinker'] = sch_to_guar['age'].apply(majority)
# %%
sch_to_guar.head()
# %%
def int_times_10(x):
    if type(x) is int:
        return 10*x
    return x
# %%
sch_to_guar.applymap(int_times_10)
# %%
