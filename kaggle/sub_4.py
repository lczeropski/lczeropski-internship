#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import sklearn.linear_model as linear_model
from autoimpute.imputations import MiceImputer, MultipleImputer, SingleImputer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

#%%

train = pd.read_csv('/Users/lczeropski/Documents/repos/lczeropski-internship/kaggle/home-data-for-ml-course/train.csv')
test = pd.read_csv('/Users/lczeropski/Documents/repos/lczeropski-internship/kaggle/home-data-for-ml-course/test.csv')
id = test.Id
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)
# %%

#%%
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = np.log(train['SalePrice']))
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
m, b = np.polyfit(train['GrLivArea'], np.log(train['SalePrice']), 1)
plt.plot(train['GrLivArea'], m*train['GrLivArea'] + b)
plt.show()
#%%
train = train[train.GrLivArea < 4000]
train.reset_index(drop=True, inplace=True)
y = train.SalePrice
X = train.drop(['SalePrice'], axis=1)
# %%
missing_data = {X.columns[i]: round(v/X.shape[0]*100,2) for i,v in enumerate(X.isna().sum()) if v>0}
missing_data = dict(sorted(missing_data.items(), key=lambda item: item[1]))
for k in missing_data.keys():
    missing_data[k] = str(missing_data[k]) + "%"

keys = list(missing_data.keys())

vals = [float(missing_data[k][:-1]) for k in keys]
plt.figure(figsize=(30,8))
sns.barplot(x=keys, y=vals)

# %%
missing_data
# %%
def fix_data(d):
    d["HasPool"] = d['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    d['has2ndfloor'] = d['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    d['hasgarage'] = d['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    d['hasbsmt'] = d['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    d['hasfireplace'] = d['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
    d['YrBltAndRemod']=d['YearBuilt']+d['YearRemodAdd']
    d["LotFrontage"].fillna(0,inplace = True)
    d['TotalSF']=d['TotalBsmtSF'] + d['1stFlrSF'] + d['2ndFlrSF']
    d['Total_porch_sf'] = (d['OpenPorchSF'] + d['3SsnPorch'] +
                              d['EnclosedPorch'] + d['ScreenPorch'] +
                              d['WoodDeckSF'])
    d.drop(['PoolQC','Utilities','YearRemodAdd',
            'Street','MiscFeature','YearBuilt'
            ],axis = 1,inplace=True)
    
#%%
fix_data(X)
fix_data(test)
#%%
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=1)
categorical_cols = [cname for cname in X_train_full.columns if 
                    #X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = test[my_cols]
#%%
def output_preds(pre):
    output = pd.DataFrame({'Id': id,
                       'SalePrice': pre})
    output.to_csv('submission.csv', index=False)

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
def score_pipe(s=100,step=100,end=3000):
    res=[]
    o = 0
    m = np.inf
    for i in list(range(s,end,step)):
        mod = XGBRegressor(learning_rate=0.01,n_estimators=i,
                                        max_depth=6, min_child_weight=0,
                                        gamma=0, subsample=0.6,
                                        colsample_bytree=0.7,
                                        objective='reg:tweedie', nthread=-1,
                                        eval_metric='poisson-nloglik',
                                        #scale_pos_weight=1,
                                        seed=27,
                                        reg_alpha=0,tree_method = 'exact')
        my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', mod)
                                ])
        my_pipeline.fit(X_train, np.log1p(y_train))
        preds = my_pipeline.predict(X_valid)
        score = mean_absolute_error(np.log1p(y_valid), preds)
        if score<m:
            m = min(m,score)
            o = i
        res.append([score,i])
        print(res[-1])
        #scores = -1 * cross_val_score(my_pipeline, X_train, y_train,
                                #cv=5,
                                #scoring='neg_mean_absolute_error')
        #print('MAE:', score,"n: ",n,"\n",scores.mean())
    print("min error:",m,'\nfor n:',o)
    return m
#%%
def test_seed(s,n):
    res=[]
    o = 0
    m = np.inf
    for i in list(range(s)):
        mod = XGBRegressor(learning_rate=0.01,n_estimators=2600,
                                        max_depth=6, min_child_weight=0,
                                        gamma=0, subsample=0.6,
                                        colsample_bytree=0.7,
                                        objective='reg:tweedie', nthread=-1,
                                        eval_metric='poisson-nloglik',
                                        #scale_pos_weight=1,
                                        seed=i,
                                        reg_alpha=0,tree_method = 'exact')
        my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', mod)
                                ])
        my_pipeline.fit(X_train, np.log(y_train))
        preds = my_pipeline.predict(X_valid)
        score = mean_absolute_error(np.log(y_valid), preds)
        if score<m:
            m = min(m,score)
            o = i
        res.append([score,i])
        print(res[-1])
        #scores = -1 * cross_val_score(my_pipeline, X_train, y_train,
                                #cv=5,
                                #scoring='neg_mean_absolute_error')
        #print('MAE:', score,"n: ",n,"\n",scores.mean())
    print("min error:",m,'\nfor s:',o)
    return m       
   
#%%
res = score_pipe(2500,100,4000)
#0.08201142544600726
#0.08201142544600726
#%%
test_seed(50,2501)
#%% best
best = 0.07950507444556268  
res <= best
#%%  er = 13668.084987692637   reg_alpha = 0.00008 ,seed=22 , n = 3420
model = XGBRegressor(learning_rate=0.01,n_estimators=2301,
                                        max_depth=6, min_child_weight=0,
                                        gamma=0, subsample=0.6,
                                        colsample_bytree=0.7,
                                        objective='reg:tweedie', nthread=-1,
                                        eval_metric='poisson-nloglik',
                                        #scale_pos_weight=1,
                                        seed=27,
                                        reg_alpha=0,tree_method = 'exact')
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])
my_pipeline.fit(X_train, np.log1p(y_train))
preds = my_pipeline.predict(X_test)
output_preds(np.floor(np.expm1(preds),2))
print(np.mean(y)/np.mean(np.round(np.expm1(preds),2)))
#%%
# %%
np.round(np.expm1(preds),2)
# %%
#hist  0.08191571390763933 for n: 1600
#exact min error: min error: 0.07978004625624305 for n: 1900
#approx min error: min error: 0.0808893880177219 for n: 1900
########
#exact
##reg:squarederror min error: 0.0840160702456001 for n: 901
##reg:squaredlogerror min error: 0.16113824577834387 for n: 901
##reg:pseudohubererror min error: 0.08663057333730591 for n: 901
##reg:tweedie min error: 0.08217056920352044 for n: 901 #dont use "scale_pos_weight"
##reg:gamma min error: 0.0960830321085012 for n: 901
#%%

# %%
best_mod = XGBRegressor(learning_rate=0.01,n_estimators=2301,
                                        max_depth=6, min_child_weight=0,
                                        gamma=0, subsample=0.6,
                                        colsample_bytree=0.7,
                                        objective='reg:tweedie', nthread=-1,
                                        eval_metric='poisson-nloglik',
                                        #scale_pos_weight=1,
                                        seed=27,
                                        reg_alpha=0,tree_method = 'exact')
#%%
from sklearn.model_selection import GridSearchCV

mod = XGBRegressor(learning_rate=0.01,n_estimators=3000,
                                        max_depth=6, min_child_weight=0,
                                        gamma=0, subsample=0.6,
                                        colsample_bytree=0.7,
                                        objective='reg:tweedie', nthread=-1,
                                        eval_metric='poisson-nloglik',
                                        #scale_pos_weight=1,
                                        seed=27,
                                        reg_alpha=0,tree_method = 'exact')
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', mod)
                                ])

# %%
param_grid = {
    #"model__n_estimators": [i for i in range(0,1000,50)],
    "model__max_depth": [2,4,6,8],
    "model__subsample" : [i/10 for i in range(1,10)],
    "model__seed" :list(range(0,30,1))
}
# %%
search = GridSearchCV(my_pipeline, param_grid, n_jobs=-1)
search.fit(X_train, np.log1p(y_train))
# %%
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)
# %%
