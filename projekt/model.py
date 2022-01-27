"""[summary]
Module to train test model, improve it by predict threshold
Model detects user with given id (default id=0 :Codename Joe)
Model requires data transformed by data_preparation.py
Supports command line arguments
--target, type:int, default=0, target id'
--split, type:float, default=0.8, split ratio
--model, type:str, default='xgboost'
    supported models are xgboost, gradient_boost, tree, forest, ada
--predict, type:bool, default=False,
    Boolean value, True to use model to make prediction on dataset'
--dataset_path, type=pathlib.Path,
    dataset path to train model, use when dataset is not in working directory'
--dataset_pred, type=pathlib.Path,
    dataset path to make predictions, used when dataset is not in working directory
--load_model', type=pathlib.Path,
    model file path to use it in predictions,
    use when you want just prediction made by model that was already trained
Returns:
   predictions.csv: csv file with predictions, only if --predict is True
   find_user_model.sav: trained model and threshold value to improve precision/recall ratio,
    only if --load_model is None
"""
import argparse
import pathlib
import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
def cal_threshold(mod_to_improve, x_val, y_val):
    """[summary]
    Function to calculate the best threshold to improve
    precision/recall ratio
    Args:
        model (sklearn model): Model to witch improvment will be find
        x_valid (Pandas DataFrame): Validation data frame with
        y_valid (array): An array with true values of users id used to validate the prediction
    Returns:
        model (sklearn model): Model to witch optimal threshold was found
        threshold (Float): Optimal threshold value
    """
    bad_preds = mod_to_improve.predict_proba(x_val)[:,1]
    precision, recall, thresholds = metrics.precision_recall_curve(y_val, bad_preds)
    fscore = (2 * precision * recall) / (precision + recall)
    index = np.argmax(fscore)
    threshold = thresholds[index]
    return mod_to_improve, threshold
def th_predict(test, model_improved, treshold=0.5):
    """[summary]
    Function to make better predictons using found threshold
    Args:
        test (Pandas DataFrame): Data to make prediction on
        model_improved (sklearn model): model to witch threshold was found
        treshold (Float): Threshold value
    Returns:
        preds (array): An array containing improved predictions
    """
    good_preds = model_improved.predict_proba(test)
    good_preds = (good_preds[:,1] >= treshold).astype('int')
    return good_preds
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser')
    parser.add_argument('--target', type=int,default=0,
            help='target id' )
    parser.add_argument('--split', type=float,default=0.8,
            help='split ratio float ' )
    parser.add_argument('--model', type=str,default='xgboost',
            help='supported models are xgboost, gradient_boost, tree, forest, ada')
    parser.add_argument('--predict', type=bool,default=False,
            help='Boolean value, True to use model to make prediction on dataset')
    parser.add_argument('--dataset_path', type=pathlib.Path,
            help='dataset path to train model, use when dataset is not in working directory')
    parser.add_argument('--dataset_pred', type=pathlib.Path,
            help='dataset path to make predictions, used when dataset is not in working directory')
    parser.add_argument('--load_model', type=pathlib.Path,
            help='model file path to use it in predictions')
    args = parser.parse_args()
    args = vars(args)
    print(args)
    if not args['load_model']:
        if args['dataset_path']:
            TRAIN_PATH = args['dataset_path']
        else:
            TRAIN_PATH = 'train_user'+str(args['target'])+".csv"
        train = pd.read_csv(TRAIN_PATH)
        y = train["user_id"]
        y = y.apply(lambda x: 0 if x==args['target'] else 1)
        X = train.drop("user_id",axis = 1)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=args['split'],
                                        test_size=1-args['split'],random_state=10,stratify=y)
        cat_cols = [cname for cname in X_train.columns if
                                X_train[cname].dtype == "object"]
        numerical_cols = [cname for cname in X_train.columns if
                          X_train[cname].dtype in ['int64', 'float64']]
        categorical_transformer_oh = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore',sparse=False))
            ])
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler(with_mean=False))
            ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('hot', categorical_transformer_oh, cat_cols)
            ])
        models = {
            'xgboost': XGBClassifier(objective='binary:logistic',
                            n_estimators = 150,
                            eval_metric = 'auc',
                            max_depth = 3,
                            random_state = 1,
                            min_child_weight = 0.5,
                            scale_pos_weight = 1,
                            use_label_encoder=False),
            'gradient_boost': GradientBoostingClassifier(n_estimators=300, learning_rate=0.01,
            max_depth=6),
            'tree': DecisionTreeClassifier(class_weight = 'balanced', criterion = 'entropy'),
            'forest': RandomForestClassifier(class_weight = 'balanced', criterion= 'entropy'),
            'ada': AdaBoostClassifier(base_estimator = DecisionTreeClassifier(
                class_weight = 'balanced', criterion = 'entropy'))
        }
        model=models[args['model']]
        pipe = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('model', model)
                                    ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_valid)
        tn, fp, fn, tp = confusion_matrix(y_valid, preds).ravel()
        print('True negatives', tn, '\nFalse positives', fp,
              '\nFalse negatives', fn, '\nTrue positives', tp)
        print(classification_report(y_valid,preds))
        results = cal_threshold(pipe,X_valid,y_valid)
        preds = th_predict(X_valid, results[0], results[1])
        print(classification_report(y_valid, preds))
        tn, fp, fn, tp = confusion_matrix(y_valid, preds).ravel()
        print('True negatives', tn, '\nFalse positives', fp,
              '\nFalse negatives', fn, '\nTrue positives', tp)
        joblib.dump(results,'find_user_model.sav')
    else:
        results = joblib.load(args['load_model'])
    if args['predict']:
        if args['dataset_path']:
            PRE_PATH = args['dataset_pred']
        else:
            PRE_PATH = 'to_pred_user' + str(args['target']) + ".csv"
        print('making prediction on data')
        to_pred = pd.read_csv(PRE_PATH)
        predictions = th_predict(to_pred, results[0], results[1])
        print(predictions)
        pd.DataFrame(predictions).to_csv('predictions.csv',index = False)
