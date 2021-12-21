#%%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
#%%
train = pd.read_csv('/Users/lczeropski/Documents/repos/lczeropski-internship/kaggle/home-data-for-ml-course/train.csv')
test = pd.read_csv('/Users/lczeropski/Documents/repos/lczeropski-internship/kaggle/home-data-for-ml-course/test.csv')
# %%
train.head(5)
#%%
train.columns
#%%
train.describe()

#%%
train.isna().sum() 
nans = { train.columns[i]: str(round((v/train.shape[0])*100,2))+"%" for i,v in enumerate(train.isna().sum()) if v!=0}
print(nans,"\n",train[nans.keys()].dtypes)
#%%
mas_vnr_area_mean = train.describe()["MasVnrArea"]["mean"]
train.MasVnrArea = train.MasVnrArea.fillna(mas_vnr_area_mean)
#%%
train = train.dropna(axis=1)
# %%
y = train.SalePrice
features = ["LotArea", "YearBuilt","1stFlrSF", "2ndFlrSF", 'FullBath',"GarageCars",
            #"Functional",
            "PoolArea", "BedroomAbvGr","Fireplaces","OverallQual","OverallCond",
            #"KitchenQual",
            "TotRmsAbvGrd","GrLivArea","YearRemodAdd","MSSubClass","HalfBath","TotalBsmtSF","MasVnrArea",
            #"HeatingQC" 
            ]
X = train[features]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
test_X = test[features]
#%%
s = (train_X.dtypes == 'object')
object_cols = list(s[s].index)
label_X_train = train_X.copy()
label_X_valid = val_X.copy()
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(train_X[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(val_X[object_cols])
def score_model(model, X_t=label_X_train, X_v=label_X_valid, y_t=train_y, y_v=val_y):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

for i in range(0, len(models)):
    mae = score_model(models[i])
    print("Model %d MAE: %d" % (i+1, mae))
#%%
rf_model = RandomForestRegressor(random_state=20)
rf_model.fit(train_X, train_y)
#%%
model_1 = RandomForestRegressor(n_estimators=50, random_state=20)
model_2 = RandomForestRegressor(n_estimators=100, random_state=20)
model_3 = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=20)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=20)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=20)
#%%
models = [model_1 ,model_2, model_3, model_4, model_5]
#%%
def score_model(model, X_t=train_X, X_v=val_X, y_t=train_y, y_v=val_y):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

for i in range(0, len(models)):
    mae = score_model(models[i])
    print("Model %d MAE: %d" % (i+1, mae))
#%%
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(train_X[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(val_X[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = train_X.index
OH_cols_valid.index = val_X.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = train_X.drop(object_cols, axis=1)
num_X_valid = val_X.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)


for i in range(0, len(models)):
    mae = score_model(models[i],X_t = OH_X_train,X_v = OH_X_valid)
    print("Model %d MAE: %d" % (i+1, mae))
#%%
predictions = rf_model.predict(val_X)
mae = mean_absolute_error(predictions, val_y)
de = train.describe()["SalePrice"]["mean"]
print("Mean Ablosute Error :",mae)
# %%
test_X = test[features]
#%%
test_nans = { test_X.columns[i]: v for i,v in enumerate(test_X.isna().sum()) if v!=0}
print(test_nans)
#%%
kitchen_test = {test["KitchenQual"].value_counts().keys()[i] : i for i, v in enumerate(test["KitchenQual"].value_counts())}
test["KitchenQual"].replace(kitchen_test,inplace=True)
funct_test = {test["Functional"].value_counts().keys()[i] : i for i, v in enumerate(test["Functional"].value_counts())}
test["Functional"].replace(funct_test,inplace=True)
heat_test = {test["HeatingQC"].value_counts().keys()[i] : i for i, v in enumerate(test["HeatingQC"].value_counts())}
test["HeatingQC"].replace(heat_test,inplace=True)
gar_cars_mean = test_X.describe()["GarageCars"]["mean"]
tot_bsm_sf_mean = test_X.describe()["TotalBsmtSF"]["mean"]
mas_vnr_area_mean_test = test.describe()["MasVnrArea"]["mean"]
kitchenQual_mean_test = test.describe()["KitchenQual"]["mean"]
functional_mean_test = test.describe()["Functional"]["mean"]
test_X["GarageCars"] = test_X["GarageCars"].fillna(gar_cars_mean)
test_X["TotalBsmtSF"] = test_X["TotalBsmtSF"].fillna(tot_bsm_sf_mean)
test_X["MasVnrArea"] = test_X["MasVnrArea"].fillna(mas_vnr_area_mean_test)
test_X["KitchenQual"] = test_X["KitchenQual"].fillna(kitchenQual_mean_test)
test_X["Functional"] = test_X["Functional"].fillna(functional_mean_test)
#%%
test_pred = rf_model.predict(test_X)
#%%
test_pred_mod1 = model_1.predict(test_X)
# %%
output = pd.DataFrame({'Id': test.Id,
                       'SalePrice': test_pred_mod1})
output.to_csv('submission.csv', index=False)
#%%

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
#%%
X_train_full, X_valid_full, y_train, y_valid = train_test_split(train, y, train_size=0.8, test_size=0.2,random_state=1)
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]


numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]


my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
#X_train.drop('SalePrice',axis='columns',inplace = True)
#X_valid.drop('SalePrice',axis='columns',inplace = True)
# %%
numerical_transformer = SimpleImputer(strategy='constant')


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
# %%
model = RandomForestRegressor(n_estimators=100, random_state=20)
# %%
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

#%%
my_pipeline.fit(X_train, y_train)

#%%
preds = my_pipeline.predict(X_valid)


score = mean_absolute_error(y_valid, preds)
print('MAE:', score)
# %%
categorical_cols_test = [cname for cname in test.columns if test[cname].nunique() < 10 and 
                        test[cname].dtype == "object"]


numerical_cols_test = [cname for cname in test.columns if test[cname].dtype in ['int64', 'float64']]


my_cols = categorical_cols_test + numerical_cols_test
X_test_pipe = test[my_cols].copy()
#%%
numerical_transformer = SimpleImputer(strategy='constant')


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols_test),
        ('cat', categorical_transformer, categorical_cols_test)
    ])
#%%
pipe_pre = my_pipeline.predict(X_test_pipe)
# %%
output = pd.DataFrame({'Id': test.Id,
                       'SalePrice': test_pred_mod1})
output.to_csv('submission.csv', index=False)
#%%
X_test_pipe
# %%
y_train
#%% XGBOOST
from xgboost import XGBRegressor

my_model = XGBRegressor()
my_model.fit(train_X, train_y)
# %%
from sklearn.metrics import mean_absolute_error

predictions = my_model.predict(val_X)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, val_y)))
# %%
my_model2 = XGBRegressor(n_estimators=500)
my_model2.fit(train_X, train_y, 
             early_stopping_rounds=5, 
             eval_set=[(val_X, val_y)],
             verbose=False)
predictions = my_model.predict(val_X)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, val_y)))
# %%

my_model3 = XGBRegressor(n_estimators=500, learning_rate=0.05,n_jobs=4)
my_model3.fit(train_X, train_y, 
             early_stopping_rounds=5, 
             eval_set=[(val_X, val_y)],
             verbose=False)
predictions = my_model.predict(val_X)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, val_y)))
# %%
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)
#%%
train = pd.read_csv('/Users/lczeropski/Documents/repos/lczeropski-internship/kaggle/home-data-for-ml-course/train.csv')
y = train.SalePrice
train.drop("SalePrice",axis = 1,inplace=True)
X_train_full, X_valid_full, y_train, y_valid= train_test_split(train, y,train_size=0.8,test_size=0.2)
cols_with_missing = [col for col in X_train_full.columns if X_train_full[col].isnull().any()] 
X_train_full.drop(cols_with_missing, axis=1, inplace=True)
X_valid_full.drop(cols_with_missing, axis=1, inplace=True)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
my_cols = low_cardinality_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
#%%
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))
#%%
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
print(score_dataset(OH_X_train, OH_X_valid, train_y, val_y))
#%%
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

# Apply ordinal encoder to each column with categorical data
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))
# %%

# %%
minimum = 10000000
#%%
n = [50,100,150,200,250,300,350,400,450,500]
for i in n:
    my_model = XGBRegressor(n_estimators=i,learning_rate = 0.05)
    my_model.fit(drop_X_train, y_train, 
                early_stopping_rounds=5, 
                eval_set=[(drop_X_valid, y_valid)],
                verbose=False)
    predictions = my_model.predict(drop_X_valid)
    mae = mean_absolute_error(predictions, y_valid)
    minimum  = min(mae,minimum )
    print("n_estimators :",i)
    print("Mean Absolute Error: " + str(mae))

# %%
test_X = test
cols_with_missing = [col for col in test_X .columns if test_X[col].isnull().any()] 
test_X.drop(cols_with_missing, axis=1, inplace=True)
low_cardinality_cols = [cname for cname in test_X.columns if test_X[cname].nunique() < 10 and 
                        test_X[cname].dtype == "object"]
numerical_cols = [cname for cname in test_X.columns if test_X[cname].dtype in ['int64', 'float64']]
my_cols_test = low_cardinality_cols + numerical_cols
X_test = test_X[my_cols_test].copy()
#%%
X_train = X_train_full[my_cols_test].copy()
X_valid = X_valid_full[my_cols_test].copy()
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])
#%%
drop_X_test = X_test.select_dtypes(exclude=['object'])
my_model = XGBRegressor(n_estimators=200,learning_rate = 0.05)
my_model.fit(drop_X_train, y_train, 
            early_stopping_rounds=5, 
            eval_set=[(drop_X_valid, y_valid)],
            verbose=False)
predictions = my_model.predict(drop_X_test)


#%%
#%%
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(test_X[object_cols_test]))

# One-hot encoding removed index; put it back
OH_cols.index = test_X.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_test = test_X.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_test = pd.concat([num_X_test, OH_cols], axis=1)
#OH_X_test.dropna(axis=1,inplace=True)
# %%
my_model = XGBRegressor(n_estimators=150,learning_rate = 0.07)
my_model.fit(OH_X_train, train_y, 
                early_stopping_rounds=5, 
                eval_set=[(OH_X_valid, val_y)],
                verbose=False)
predictions = my_model.predict(OH_X_test)
# %%
object_cols
# %%
OH_X_test.isna().sum()
# %%
train["Electrical"].unique
# %%
