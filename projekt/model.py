
import argparse
import pathlib
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
from sklearn.ensemble import GradientBoostingClassifier

parser = argparse.ArgumentParser(description='Parser')
parser.add_argument('train', type=pathlib.Path,
                    help='train data path' )
parser.add_argument('--target', type=int,default=0,
                    help='target id ' )
parser.add_argument('--split', type=float,default=0.8,
                    help='split ratio float ' )
parser.add_argument('--model', type=int,default=0,
                    help='1 to use different model' )
parser.add_argument('--pre', type=pathlib.Path,default=None,
                    help='to predict data path' )

args = parser.parse_args()
print(vars(args))
args = vars(args)

def read_data(path):
    data = pd.read_json(path) 
    return data
train = read_data(args['train'])
target = args['target']


def data_prep(data,tgt=pd.DataFrame(None)):
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
    if tgt.shape[0]==0: 
        tgt = data[data['user_id']==target]
    else :
        tgt = tgt   
    for i in ['browser','os','locale','location']:
        tem = tgt[i].unique()
        for j in (data[i].unique()):
            if j not in tem:
                data[i].replace(to_replace = j,value = 'Other',inplace = True)
    for i in range(0,len(sites.columns)):
        tem = tgt['site'+str(i)].value_counts()[:10].index
        for j in set(data["site"+str(i)]):
            if j not in tem:
                data["site"+str(i)].replace(to_replace=j,value='Other',inplace=True)

    return data,tgt

train,tgt = data_prep(train)
y = train["user_id"]
y = y.apply(lambda x: 0 if x==target else 1)
X = train.drop("user_id",axis = 1)


X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=args['split'], test_size=1-args['split'],random_state=10,stratify=y)


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
if args['model']==0:
    model = XGBClassifier(objective='binary:logistic',
                      n_estimators = 150,
                      eval_metric = 'auc',
                      max_depth = 3,
                      random_state = 1,
                      min_child_weight = 0.5,
                      scale_pos_weight = 1,
                      use_label_encoder=False)
else:
    model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.01,
    max_depth=6)
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
if args['pre']:
    print('making prediction on data')
    to_pred= read_data((args['pre'][0]))
    test = data_prep(to_pred,tgt)[0]
    predictions = predict(test)
    print(predictions)

