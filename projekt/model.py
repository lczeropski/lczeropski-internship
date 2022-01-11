
import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


def read_data():
    data = pd.read_json(input()) #/Users/lczeropski/Documents/usersessions/dataset.json
    return data

train = read_data()

def data_prep(data):
    target = int(input())
    tgt = data[data['user_id'] == target]
    sites = pd.json_normalize(data.sites) 
    dt=data.drop('sites',axis = 1) 
    for i in range(0,len(sites.columns)):
        dt['site'+str(i)] = pd.json_normalize(sites[i]).site 
        dt['site_len'+str(i)] = pd.json_normalize(sites[i]).length 
        dt['site'+str(i)].fillna('None',inplace=True) 
        dt['site_len'+str(i)].fillna(0,inplace = True) 
    data = dt 
    data["time"] = pd.to_timedelta(data["time"])
    data["time"] = data["time"].dt.components["hours"]*60 + data["time"].dt.components["minutes"] 
    data['date'] = pd.to_datetime(data['date']).view(np.int64) 
    data["gender"].replace({'m':0,'f':1},inplace = True)    
    for i in ['browser','os','locale','location']:
        tem = tgt[i].unique()
        for j in (data[i].unique()):
            if j not in tem:
                data[i].replace(to_replace = j,value = 'Other',inplace = True)
    tgt = data[data['user_id']==target]
    for i in range(0,len(sites.columns)):
        tem = tgt['site'+str(i)].value_counts()[:10].index
        for j in set(data["site"+str(i)]):
            if j not in tem:
                data["site"+str(i)].replace(to_replace=j,value='Other',inplace=True)
    y = data["user_id"]
    y = y.apply(lambda x: -1 if x==target else x)
    y = y.apply(lambda x: 0 if x>=0  else 1)
    X = data.drop("user_id",axis = 1)
    return X,y


X,y = data_prep(train)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=10,stratify=y)


cat_cols = [cname for cname in X_train.columns if  
                        X_train[cname].dtype == "object"] 
numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]

categorical_transformer_oh = Pipeline(steps=[ 
    ('onehot', OneHotEncoder(handle_unknown='ignore',sparse=False))
])
numerical_transformer = SimpleImputer(strategy='mean')
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('hot', categorical_transformer_oh, cat_cols)
    ])
model = XGBClassifier(objective='binary:logistic',
                      n_estimators = 150,
                      eval_metric = 'auc',
                      max_depth = 3,
                      random_state = 1,
                      min_child_weight = 0.5,
                      scale_pos_weight = 1,
                      use_label_encoder=False)
pipe = Pipeline(steps=[('preprocessor', preprocessor),
                       ('standardscaler', StandardScaler(with_mean=False)),
                              ('model', model)
                             ])
pipe.fit(X_train, y_train)

preds = pipe.predict(X_valid)
tn, fp, fn, tp = confusion_matrix(y_valid,preds).ravel()
print('True negatives',tn,'\nFalse positives',fp,'\nFalse negatives',fn,'\nTrue positives',tp)
print(classification_report(y_valid,preds))

preds = pipe.predict_proba(X_valid)[:,1]
precision, recall, thresholds = metrics.precision_recall_curve(y_valid,preds)
fscore = (2 * precision * recall) / (precision + recall)
ix = np.argmax(fscore)

def predict(data,model = pipe):
    threshold =thresholds[ix]
    preds = model.predict_proba(data)
    preds = (preds[:,1] >= threshold).astype('int')
    return preds
preds = predict(X_valid)
print(classification_report(y_valid,preds))
tn, fp, fn, tp = confusion_matrix(y_valid,preds).ravel()
print('True negatives',tn,'\nFalse positives',fp,'\nFalse negatives',fn,'\nTrue positives',tp)

joblib.dump(pipe,'find_user_model')

