#%%
import numpy as np 
import pandas as pd 

from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from autoimpute.imputations import SingleImputer, MultipleImputer, MiceImputer
#%%
train = pd.read_csv('/Users/lczeropski/Documents/repos/lczeropski-internship/kaggle/home-data-for-ml-course/train.csv')
test = pd.read_csv('/Users/lczeropski/Documents/repos/lczeropski-internship/kaggle/home-data-for-ml-course/test.csv')
y = train.SalePrice
X = train.drop(['SalePrice'], axis=1)
#%%
missing_data = {train.columns[i]: round(v/train.shape[0]*100,2) for i,v in enumerate(train.isna().sum()) if v>0}
missing_data = dict(sorted(missing_data.items(), key=lambda item: item[1]))
for k in missing_data.keys():
    missing_data[k] = str(missing_data[k]) + "%"

keys = list(missing_data.keys())

vals = [float(missing_data[k][:-1]) for k in keys]
plt.figure(figsize=(30,8))
sns.barplot(x=keys, y=vals)
# %%
X.shape[1]==test.shape[1]
#%%
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=1)
# %% train data
categorical_cols = [cname for cname in X_train_full.columns if 
                    #X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
#%%
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
#%%
missing_data_train = {X_train.columns[i]: round(v/X_train.shape[0]*100,2) for i,v in enumerate(X_train.isna().sum()) if v>0}
missing_data_train = dict(sorted(missing_data_train.items(), key=lambda item: item[1]))
for k in missing_data_train.keys():
    missing_data_train[k] = str(missing_data_train[k]) + "%"

keys = list(missing_data_train.keys())

vals = [float(missing_data_train[k][:-1]) for k in keys]
plt.figure(figsize=(30,8))
sns.barplot(x=keys, y=vals)
#%% test data
X_test = test[my_cols]
#%%
def output_preds(pre):
    output = pd.DataFrame({'Id': test.Id,
                       'SalePrice': pre})
    output.to_csv('submission.csv', index=False)
#%%
si = SingleImputer() # pass through data once
mi = MultipleImputer() # pass through data multiple times
mice = MiceImputer() # pass through data multiple times and iteratively optimize imputations in each column
# %% how to deal with missing num values
numerical_transformer = SimpleImputer(strategy='mean')
# %% deal with missing cat values
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# %% preprocessor construction
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
#%%
model2 = XGBRegressor(learning_rate=0.01,n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:squarederror', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)
#%%
#model1 = XGBRegressor(n_estimators=n)
# %%
def score_pipe(n=100):
    mod = XGBRegressor(learning_rate=0.01,n_estimators=n,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:squarederror', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00008)
    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', mod)
                             ])
    my_pipeline.fit(X_train, y_train)
    preds = my_pipeline.predict(X_valid)
    score = mean_absolute_error(y_valid, preds)
    scores = -1 * cross_val_score(my_pipeline, X_train, y_train,
                              cv=5,
                              scoring='neg_mean_absolute_error')
    print('MAE:', score,"n: ",n,"\n",scores.mean())
    #print(score)
    return [score, n]
    
#%%
nums = list(range(3000,4000,20)) 
#%%
res = []
for i in nums:
    res.append(score_pipe(i))
print("done")
#%%  
res
    
# n = 250 l= 0.06
# %% 13812.773919092466, n= 3440
#%%  er = 13668.084987692637   reg_alpha = 0.00008 ,seed=22 , n = 3420
model = XGBRegressor(learning_rate=0.01,n_estimators=3080,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:squarederror', nthread=-1,
                                     scale_pos_weight=1, seed=22,
                                     reg_alpha=0.00001)
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])
my_pipeline.fit(X_train, y_train)
preds = my_pipeline.predict(X_test)
output_preds(preds) #15944.07903 , 15961.44058 14265.13001
# best 14031.79219



