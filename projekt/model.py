
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
def cal_threshold(model,X_val,y_val):
    preds = model.predict_proba(X_val)[:,1]
    precision, recall, thresholds = metrics.precision_recall_curve(y_val,preds)
    fscore = (2 * precision * recall) / (precision + recall)
    ix = np.argmax(fscore)
    threshold =thresholds[ix]
    return model,threshold
def th_predict(test,model,th):
    preds = model.predict_proba(test)
    preds = (preds[:,1] >= th).astype('int')
    return preds
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser')
    parser.add_argument('--target', type=int,default=0,
                        help='target id' )
    parser.add_argument('--split', type=float,default=0.8,
                        help='split ratio float ' )
    parser.add_argument('--model', type=str,default='xgboost',
                        help='supported models are xgboost and gradient_boost' )
    parser.add_argument('--predict', type=bool,default=False,
                        help='Boolean value, True to use model to make prediction on dataset' )
    parser.add_argument('--dataset_path', type=pathlib.Path,
                        help='dataset path to train model, use when dataset is not in working directory' )
    parser.add_argument('--dataset_pred', type=pathlib.Path,
                        help='dataset path to make predictions, used when dataset is not in working directory' )

    args = parser.parse_args()
    print(vars(args))
    args = vars(args)

    if args['dataset_path']:
        train_path = args['dataset_path']
    else:
        train_path = 'train_user'+str(args['target'])
    train = pd.read_csv(train_path)
    y = train["user_id"]
    y = y.apply(lambda x: 0 if x==args['target'] else 1)
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
    if args['model']=='xgboost':
        model = XGBClassifier(objective='binary:logistic',
                        n_estimators = 150,
                        eval_metric = 'auc',
                        max_depth = 3,
                        random_state = 1,
                        min_child_weight = 0.5,
                        scale_pos_weight = 1,
                        use_label_encoder=False)
    elif args['model']=='gradient_boost':
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
    results = cal_threshold(pipe,X_valid,y_valid)
    


    preds = th_predict(X_valid,results[0],results[1])
    print(classification_report(y_valid,preds))
    tn, fp, fn, tp = confusion_matrix(y_valid,preds).ravel()
    print('True negatives',tn,'\nFalse positives',fp,'\nFalse negatives',fn,'\nTrue positives',tp)

    joblib.dump(pipe,'find_user_model')
    if args['predict']:
        if args['dataset_path']:
            pre_path = args['dataset_pred']
        else:
            pre_path = 'to_pred_user'+str(args['target'])
        print('making prediction on data')
        to_pred=pd.read_csv(pre_path)
        predictions = th_predict(to_pred,results[0],results[1])
        print(predictions)
        pd.DataFrame(predictions).to_csv('predictions')

