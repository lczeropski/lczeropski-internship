#%%
import pandas as pd
# %%
raw_data_1 = {
        'subject_id': ['1', '2', '3', '4', '5'],
        'first_name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'], 
        'last_name': ['Anderson', 'Ackerman', 'Ali', 'Aoni', 'Atiches']}

raw_data_2 = {
        'subject_id': ['4', '5', '6', '7', '8'],
        'first_name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'], 
        'last_name': ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']}

raw_data_3 = {
        'subject_id': ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11'],
        'test_id': [51, 15, 15, 61, 16, 14, 15, 1, 61, 16]}
# %%
data1 = pd.DataFrame(raw_data_1, columns= ['subject_id', 'first_name', 'last_name'])
data2 = pd.DataFrame(raw_data_2, columns= ['subject_id', 'first_name', 'last_name'])
data3 = pd.DataFrame(raw_data_3, columns= ['subject_id', 'test_id'])
# %%
data1,data2,data3
# %%
all_data = pd.concat([data1,data2])
# %%
all_data
# %%
all_data_col = pd.concat([data1,data2],axis = 1)
# %%
all_data_col
# %%
data3
# %%
pd.merge(all_data, data3, on='subject_id')
# %%
pd.merge(data1, data2, on='subject_id', how='inner') #same 'subject_id' on both data1 and data2
# %%
#matching records from both sides where available
pd.merge(data1, data2, on='subject_id', how='outer')
# %%
import numpy as np

# %%
s1 = pd.Series(np.random.randint(1, 5, 100, dtype='int32'))
s2 = pd.Series(np.random.randint(1, 4, 100, dtype='int32'))
s3 = pd.Series(np.random.randint(10000, 30001, 100, dtype='int32'))

print(s1, s2, s3)
# %%
data = pd.concat([s1,s2,s3],axis = 1)
# %%
data
# %%
data.rename(columns = {0: 'bedrs', 1: 'bathrs', 2: 'price_sqr_meter'},inplace = True)
# %%
data
# %%
bigcolumn = pd.concat([s1,s2,s3],axis = 0)
# %%
bigcolumn = bigcolumn.to_frame()
# %%
print(type(bigcolumn))
# %%
bigcolumn
# %%
len(bigcolumn)
# %%
bigcolumn.reset_index(drop = True, inplace = True)
# %%
bigcolumn
# %%
