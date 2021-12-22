#%%
import numpy as np 
import pandas as pd 
from datetime import datetime
from scipy.stats import skew  
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
#%%
train = pd.read_csv('/Users/lczeropski/Documents/repos/lczeropski-internship/kaggle/home-data-for-ml-course/train.csv')
test = pd.read_csv('/Users/lczeropski/Documents/repos/lczeropski-internship/kaggle/home-data-for-ml-course/test.csv')
#%%
y = train.SalePrice
X = train.drop(['SalePrice'], axis=1)
# %%
X.shape[1]==test.shape[1]
# %%
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=1)
# %% test NaN
cols_with_missing_test = [col for col in test.columns if test[col].isnull().any()] 
test.drop(cols_with_missing_test, axis=1, inplace=True)
good_cols = test.columns
# %% train NaN col from test
X_train_full = X_train_full[good_cols]
X_valid_full = X_valid_full[good_cols]
# %%
X_train_full.shape[1],test.shape[1] # Num of cols is equal
# %%
low_cardinality_cols_train = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]
low_cardinality_cols_test = [cname for cname in test.columns if test[cname].nunique() < 10 and 
                        test[cname].dtype == "object"]
# %%
len(low_cardinality_cols_test) == len(low_cardinality_cols_train) #True
# %%
numerical_cols_train = [cname for cname in X_train_full.columns if 
                  X_train_full[cname].dtype in ['int64', 'float64']]
numerical_cols_test = [cname for cname in test.columns if 
                  test[cname].dtype in ['int64', 'float64']]

# %%
len(numerical_cols_test) == len(numerical_cols_train) #True
# %%
final_cols = low_cardinality_cols_test + numerical_cols_test
# %%
X_train = X_train_full[final_cols].copy()
X_valid = X_valid_full[final_cols].copy()
X_test = test[final_cols].copy()
#%%
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)
#%%
n = [50,100,150,200,250]
def score(fit1,fit2,val1,val2,l=0.05):
    m = [0,np.inf]
    
    for i in n:
        my_model = XGBRegressor(n_estimators=i,learning_rate = l)
        my_model.fit(fit1, fit2, 
                    early_stopping_rounds=5, 
                    eval_set=[(val1, val2)],
                    verbose=False)
        predictions = my_model.predict(val1)
        mae = mean_absolute_error(predictions, val2)
        m = [i,min(m[1],mae)]
    return m,l
# %% #DROP method
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])
drop_X_test = X_test.select_dtypes(exclude=['object'])
#%%
score(drop_X_train,y_train,drop_X_valid,y_valid,0.07) #17122.301904965752 n=150


# %%
my_model = XGBRegressor(n_estimators=150,learning_rate = 0.07)
my_model.fit(drop_X_train, y_train, 
                early_stopping_rounds=5, 
                eval_set=[(drop_X_valid, y_valid)],
                verbose=False)
predictions = my_model.predict(drop_X_test)
# %%
def output_preds(pre):
    output = pd.DataFrame({'Id': test.Id,
                       'SalePrice': pre})
    output.to_csv('submission.csv', index=False)
# %%
output_preds(predictions) #17798.38151
# %% Ordinal Ecoding
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()
label_X_test = X_test.copy()
# Apply ordinal encoder to each column with categorical data
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
label_X_valid[object_cols] = ordinal_encoder.fit_transform(X_valid[object_cols])
label_X_test[object_cols] = ordinal_encoder.fit_transform(X_test[object_cols])

score(label_X_train,y_train,label_X_valid,y_valid,0.05) #n=200 err = 16665.346479023974
# %%
def pre(n,l,tre1,val1,p):
    my_model = XGBRegressor(n_estimators=n,learning_rate = l)
    my_model.fit(tre1, y_train, 
                early_stopping_rounds=5, 
                eval_set=[(val1, y_valid)],
                verbose=False)
    result = my_model.predict(p)
    return result
# %%
predictions = pre(200,0.05,label_X_train,label_X_valid,label_X_test)
# %%
output_preds(predictions) #19073.53632
# %%ONE-HOT
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))
OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[object_cols]))
# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index
OH_cols_test.index = X_test.index
# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)
num_X_test = X_test.drop(object_cols,axis = 1)
# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)
# %%
score(OH_X_train,y_train,OH_X_valid,y_valid,0.04) #n = 200, err = 16536.996561964897
# %%
predictions = pre(200,0.04,OH_X_train,OH_X_valid,OH_X_test)
# %%
output_preds(predictions) #18789.93710

# %%%
## More preprocessing
train = pd.read_csv('/Users/lczeropski/Documents/repos/lczeropski-internship/kaggle/home-data-for-ml-course/train.csv')
test = pd.read_csv('/Users/lczeropski/Documents/repos/lczeropski-internship/kaggle/home-data-for-ml-course/test.csv')
#%%
y = train.SalePrice
X = train.drop(['SalePrice'], axis=1)
# %%
X.shape[1]==test.shape[1]
# %%
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=1)
# %% train data
categorical_cols = [cname for cname in X_train_full.columns if 
                    #X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
# %%
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
#%% test data
X_test = test[my_cols]
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

# %%
def score_pipe(n=100,l=0.05,r=0):
    model = XGBRegressor(n_estimators=n,learning_rate = l, random_state = r)
    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])
    my_pipeline.fit(X_train, y_train)
    preds = my_pipeline.predict(X_valid)
    score = mean_absolute_error(y_valid, preds)
    #scores = -1 * cross_val_score(my_pipeline, X_train, y_train,
                              #cv=5,
                              #scoring='neg_mean_absolute_error')
    #print('MAE:', score,"n: ",n,"l: ",l,"\n",scores.mean())
    return [score, n]
    
#%%
nums = list(range(100,3000,50)) 
#%%
res = []
for i in nums:
    res.append(score_pipe(i,0.06,3))
#%%  
print(min(res[0]))
    
# n = 250 l= 0.06
# %%
model = XGBRegressor(n_estimators=200,learning_rate = 0.07,random_state = 24)
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])
my_pipeline.fit(X_train, y_train)
preds = my_pipeline.predict(X_test)
output_preds(preds) #15944.07903 , 15961.44058

# %% KAGGLE BEST
train = pd.read_csv('/Users/lczeropski/Documents/repos/lczeropski-internship/kaggle/home-data-for-ml-course/train.csv')
test = pd.read_csv('/Users/lczeropski/Documents/repos/lczeropski-internship/kaggle/home-data-for-ml-course/test.csv')

y = train.SalePrice
num_columns = [f for f in train.columns if train.dtypes[f] != 'object']
num_columns.remove('SalePrice')
num_columns.remove('Id')
cat_columns = [f for f in train.columns if train.dtypes[f] == 'object']

missing_data = {train.columns[i]: round(v/train.shape[0]*100,2) for i,v in enumerate(train.isna().sum()) if v>0}
missing_data = dict(sorted(missing_data.items(), key=lambda item: item[1]))
for k in missing_data.keys():
    missing_data[k] = str(missing_data[k]) + "%"

keys = list(missing_data.keys())

vals = [float(missing_data[k][:-1]) for k in keys]
#%%
plt.figure(figsize=(30,8))
sns.barplot(x=keys, y=vals)
# %%
sns.distplot(y, kde=False, fit=stats.johnsonsu)
#%%
sns.distplot(y, kde=False, fit=stats.norm)
#%%
sns.distplot(y, kde=False, fit=stats.lognorm)


# %%
def encode(frame, feature):
    ordering = pd.DataFrame()
    ordering['val'] = frame[feature].unique()
    ordering.index = ordering.val
    ordering['spmean'] = frame[[feature, 'SalePrice']].groupby(feature).mean()['SalePrice']
    ordering = ordering.sort_values('spmean')
    ordering['ordering'] = range(1, ordering.shape[0]+1)
    ordering = ordering['ordering'].to_dict()
    
    for cat, o in ordering.items():
        frame.loc[frame[feature] == cat, feature+'_E'] = o
    
qual_encoded = []
for q in cat_columns:  
    encode(train, q)
    qual_encoded.append(q+'_E')
print(qual_encoded)
# %%
def spearman(frame, features):
    spr = pd.DataFrame()
    spr['feature'] = features
    spr['spearman'] = [frame[f].corr(frame['SalePrice'], 'spearman') for f in features]
    spr = spr.sort_values('spearman')
    plt.figure(figsize=(6, 0.25*len(features)))
    sns.barplot(data=spr, y='feature', x='spearman', orient='h')
    
features = num_columns + qual_encoded
# %%
corr = train[num_columns+['SalePrice']].corr()
sns.heatmap(corr)
# %%
corr = train[qual_encoded+['SalePrice']].corr()
sns.heatmap(corr)
# %%
corr = pd.DataFrame(np.zeros([len(num_columns)+1, len(qual_encoded)+1]), index=num_columns+['SalePrice'], columns=qual_encoded+['SalePrice'])
for q1 in num_columns+['SalePrice']:
    for q2 in qual_encoded+['SalePrice']:
        corr.loc[q1, q2] = train[q1].corr(train[q2])
sns.heatmap(corr)
# %%
features = num_columns + qual_encoded
model = TSNE(n_components=2, random_state=0, perplexity=50)
X = train[features].fillna(0.).values
tsne = model.fit_transform(X)

std = StandardScaler()
s = std.fit_transform(X)
pca = PCA(n_components=30)
pca.fit(s)
pc = pca.transform(s)
kmeans = KMeans(n_clusters=5)
kmeans.fit(pc)

fr = pd.DataFrame({'tsne1': tsne[:,0], 'tsne2': tsne[:, 1], 'cluster': kmeans.labels_})
sns.lmplot(data=fr, x='tsne1', y='tsne2', hue='cluster', fit_reg=False)
print(np.sum(pca.explained_variance_ratio_))
# %%
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)
# %%
train = train[train.GrLivArea < 4500]
train.reset_index(drop=True, inplace=True)
train["SalePrice"] = np.log1p(train["SalePrice"])
y = train['SalePrice'].reset_index(drop=True)
# %%
train_features = train.drop(['SalePrice'], axis=1)
test_features = test
features = pd.concat([train_features, test_features]).reset_index(drop=True)
# %%
features.head(10)
# %%
objects = []
for i in features.columns:
    if features[i].dtype == object:
        objects.append(i)
features.update(features[objects].fillna('None'))

features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics = []
for i in features.columns:
    if features[i].dtype in numeric_dtypes:
        numerics.append(i)
features.update(features[numerics].fillna(0))
# %%
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics2 = []
for i in features.columns:
    if features[i].dtype in numeric_dtypes:
        numerics2.append(i)
skew_features = features[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

for i in skew_index:
    features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))
# %%
features = features.drop(['Utilities', 'Street', 'PoolQC',], axis=1)

features['YrBltAndRemod']=features['YearBuilt']+features['YearRemodAdd']
features['TotalSF']=features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']

features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +
                                 features['1stFlrSF'] + features['2ndFlrSF'])

features['Total_Bathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) +
                               features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))

features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +
                              features['EnclosedPorch'] + features['ScreenPorch'] +
                              features['WoodDeckSF'])
# %%
features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
# %%
final_features = pd.get_dummies(features).reset_index(drop=True)
final_features.shape
# %%
X = final_features.iloc[:len(y), :]
X_sub = final_features.iloc[len(y):, :]
X.shape, y.shape, X_sub.shape
# %%
outliers = [30, 88, 462, 631, 1322]
X = X.drop(X.index[outliers])
y = y.drop(y.index[outliers])

overfit = []
for i in X.columns:
    counts = X[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X) * 100 > 99.94:
        overfit.append(i)

overfit = list(overfit)
X = X.drop(overfit, axis=1)
X_sub = X_sub.drop(overfit, axis=1)
# %%
X.shape, y.shape, X_sub.shape
# %%
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))
    return (rmse)
# %%
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
# %%
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds))
elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))                                
svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003,))
# %%
gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt',
                                min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42)   
# %%
lightgbm = LGBMRegressor(objective='regression', 
                                       num_leaves=4,
                                       learning_rate=0.01, 
                                       n_estimators=5000,
                                       max_bin=200, 
                                       bagging_fraction=0.75,
                                       bagging_freq=5, 
                                       bagging_seed=7,
                                       feature_fraction=0.2,
                                       feature_fraction_seed=7,
                                       verbose=-1,
                                       )
# %%
xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)
# %%
stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr, xgboost, lightgbm),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)
# %%
import warnings
warnings.filterwarnings('ignore')
score = cv_rmse(ridge)
score = cv_rmse(lasso)
print(f"LASSO: {score.mean()} {score.std()}\n", datetime.now(), )

score = cv_rmse(elasticnet)
print(f"elastic net: {score.mean()} {score.std()}\n", datetime.now(), )

score = cv_rmse(svr)
print(f"SVR: {score.mean()} {score.std()}\n", datetime.now(), )

score = cv_rmse(lightgbm)
print(f"lightgbm: {score.mean()} {score.std()}\n", datetime.now(), )

score = cv_rmse(gbr)
print(f"gbr: {score.mean()} {score.std()}\n", datetime.now(), )

score = cv_rmse(xgboost)
print(f"xgboost: {score.mean()} {score.std()}\n", datetime.now(), )
# %%
print('START Fit')

print('stack_gen')
stack_gen_model = stack_gen.fit(np.array(X), np.array(y))

print('elasticnet')
elastic_model_full_data = elasticnet.fit(X, y)

print('Lasso')
lasso_model_full_data = lasso.fit(X, y)

print('Ridge')
ridge_model_full_data = ridge.fit(X, y)

print('Svr')
svr_model_full_data = svr.fit(X, y)

print('GradientBoosting')
gbr_model_full_data = gbr.fit(X, y)

print('xgboost')
xgb_model_full_data = xgboost.fit(X, y)

print('lightgbm')
lgb_model_full_data = lightgbm.fit(X, y)
#%%
def blend_models_predict(X):
    return ((0.1 * elastic_model_full_data.predict(X)) + \
            (0.05 * lasso_model_full_data.predict(X)) + \
            (0.1 * ridge_model_full_data.predict(X)) + \
            (0.1 * svr_model_full_data.predict(X)) + \
            (0.1 * gbr_model_full_data.predict(X)) + \
            (0.15 * xgb_model_full_data.predict(X)) + \
            (0.1 * lgb_model_full_data.predict(X)) + \
            (0.3 * stack_gen_model.predict(np.array(X))))
# %%
print('RMSLE score on train data:')
print(rmsle(y, blend_models_predict(X)))
# %%
# %%
print('Predict submission')
submission = pd.read_csv("/Users/lczeropski/Documents/repos/lczeropski-internship/kaggle/home-data-for-ml-course/sample_submission.csv")
#%%
submission.iloc[:,1] = np.floor(np.expm1(blend_models_predict(X_sub)))
submission.to_csv("submission.csv", index=False)
# %%
submission