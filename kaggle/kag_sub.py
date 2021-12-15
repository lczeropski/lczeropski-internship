#%%
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
#%%
train = pd.read_csv('/Users/lczeropski/Documents/repos/lczeropski-internship/kaggle/home-data-for-ml-course/train.csv')
test = pd.read_csv('/Users/lczeropski/Documents/repos/lczeropski-internship/kaggle/home-data-for-ml-course/test.csv')
# %%
train.head(5)
#%%
train.columns
#%%
summary = train.describe()
#%%
train.isna().sum() 
nans = { train.columns[i]: v for i,v in enumerate(train.isna().sum()) if v!=0}
nans
#%%
train = train.dropna(axis=1)
#%%
train.columns
#%%
train.dtypes
# %%
y = train.SalePrice
features = ["LotArea", "YearBuilt","1stFlrSF", "2ndFlrSF", 'FullBath',"GarageCars",
            "PoolArea", "BedroomAbvGr","Fireplaces","OverallQual","OverallCond",
            "TotRmsAbvGrd","GrLivArea","YearRemodAdd","MSSubClass","HalfBath","TotalBsmtSF",]
X = train[features]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
test_X = test[features]
test_X = test_X.fillna(0)
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
predictions = rf_model.predict(val_X)
mae = mean_absolute_error(predictions, val_y)
print("Mean Ablosute Error :",mae)
# %%
rf_model.predict(test_X)
# %%
