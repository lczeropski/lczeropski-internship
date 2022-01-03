#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import sklearn.linear_model as linear_model
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
# %%
train = pd.read_json('/Users/lczeropski/Documents/usersessions/dataset.json')
test = pd.read_json('/Users/lczeropski/Documents/usersessions/verify.json')
#%%
train[train["user_id"]==0]
# %%
train.dtypes
#%%
#%%
def fix_data(x):
    sites = pd.json_normalize(x.sites)
    dt=x.drop('sites',axis = 1)
    for i in range(0,len(sites.columns)):
        dt['site'+str(i)] = pd.json_normalize(sites[i]).site
        dt['site_len'+str(i)] = pd.json_normalize(sites[i]).length
        dt['site'+str(i)].fillna('None',inplace=True)
        dt['site_len'+str(i)].fillna(0,inplace = True)
    x = dt
    x["time"] = pd.to_timedelta(x["time"])
    con = x["time"].dt.components
    con["seconds"].sum()
    x["time"] = con["hours"]*60 + con["minutes"] 
    x['time'].head()
    x['date'] = pd.to_datetime(x['date']).view(np.int64)
    x["gender"].replace({'m':0,'f':1},inplace = True)
    return x
#%%
train = fix_data(train)
#%%
test = fix_data(test)
# %%
y = train["user_id"]
X = train.drop("user_id",axis = 1)

#%%
y[y>0]=1
#%%
y.head()
# %%
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=10)
categorical_cols = [cname for cname in X_train_full.columns if 
                    #X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
#%%
X_test = test[my_cols]
#%%
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

# %%
model = XGBRegressor(learning_rate=0.01,n_estimators=2000,
                                        max_depth=6, min_child_weight=0,
                                        gamma=0, subsample=0.6,
                                        colsample_bytree=0.7,
                                        objective='binary:hinge', nthread=-1,
                                        #eval_metric='poisson-nloglik',
                                        #scale_pos_weight=1,
                                        seed=27,
                                        reg_alpha=0,tree_method = 'exact')
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])
my_pipeline.fit(X_train, y_train)
preds = my_pipeline.predict(X_valid)
accuracy_score(y_valid,preds)
#%%
preds = my_pipeline.predict(X_test)
# %%
param_grid = {
    #"model__n_estimators": [i for i in range(0,1000,50)],
    "model__max_depth": [2,4,6,8],
    "model__subsample" : [i/10 for i in range(1,10)],
    "model__seed" :list(range(0,30,1))
}
# %%
search = GridSearchCV(my_pipeline, param_grid, n_jobs=-1)
search.fit(X_train,y_train)
# %%
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)# %%


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer( use_idf=False, smooth_idf=True, sublinear_tf=False,binary=True)
def vectorize(x):
    for i in range(0,13):
        x["site"+str(i)] = list(vectorizer.fit_transform(x["site"+str(i)]).todense()) #list
        
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer( use_idf=False, smooth_idf=True, sublinear_tf=False,binary=True)
def vectorize(x):
    sites = pd.DataFrame()
    for i in range(0,13):
        encodings = vectorizer.fit_transform(x["site"+str(i)])
        sites["site"+str(i)] = list(encodings.toarray())
        x["site"+str(i)] =np.array(sites["site"+str(i)].tolist())