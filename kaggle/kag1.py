#%%
import pandas as pd
# %%
melbourne_file_path = '/Users/lczeropski/Documents/repos/lczeropski-internship/kaggle/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
melbourne_data.describe()
# %%
summary = melbourne_data.describe()
# %%
summary.loc['min']['Rooms']
# %%
melbourne_data.columns
melbourne_data.head()
# %%
melbourne_data = melbourne_data.dropna(axis=0)
# %%
y = melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
# %%
X.describe()
# %%
X.head()
## 1. Define: What type of model will it be? A decision tree? Some other type of model? 
# Some other parameters of the model type are specified too.
## 2. Fit: Capture patterns from provided data. This is the heart of modeling.
## 3. Predict: Just what it sounds like
## 4. Evaluate: Determine how accurate the model's predictions are.
#%%
from sklearn.tree import DecisionTreeRegressor
# %%
# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)
# %%
melbourne_model.fit(X, y)
#%%
print(X.head()) # For 5 houses
#%%
print(melbourne_model.predict(X.head())) # predictions are as follow
# %%
## error=actual−predicted
# Data Loading Code Hidden Here
import pandas as pd

# Load data
melbourne_data = pd.read_csv(melbourne_file_path) 
# Filter rows with missing price values
filtered_melbourne_data = melbourne_data.dropna(axis=0)
# Choose target and features
y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]

from sklearn.tree import DecisionTreeRegressor
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(X, y)
#%%
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)
# %%
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(train_X, train_y)
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
# %%
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
# %%
# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
## 500 is optimal
# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))

# %%
