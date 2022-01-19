#%%
import pandas as pd
import numpy as np
import argparse
import pathlib
class Data_prep:
    def __init__(self,path,target_id):
        self.path = path
        self.data = pd.read_json(self.path)
        self.target = target_id
    def _flatten(self):
        sites = pd.json_normalize(self.data.sites)
        self.lensi = len(sites.columns)
        dt=self.data.drop('sites',axis = 1) 
        for i in range(0,len(sites.columns)):
            dt['site'+str(i)] = pd.json_normalize(sites[i]).site 
            dt['site_len'+str(i)] = pd.json_normalize(sites[i]).length 
            dt['site'+str(i)].fillna('None',inplace=True) 
            dt['site_len'+str(i)].fillna(0,inplace = True) 
        self.data = dt 
        self.data["time"] = pd.to_timedelta(self.data["time"])
        self.data["time"] = self.data["time"].dt.components["hours"]*60 + self.data["time"].dt.components["minutes"] 
        self.data['date'] = pd.to_datetime(self.data['date']).view(np.int64) 
        self.data["gender"].replace({'m':0,'f':1},inplace = True)
    def get_target(self):
        return self.target
    def set_target(self,target_id):
        self.target = target_id
    def get_pattern(self):
        return repr(self.pattern)
    def _generate_pattern(self):
        separeted = self.data[self.data['user_id']==self.target]
        separeted.to_csv('user'+'_'+str(self.target))
    def _set_pattern(self):
        self.pattern = pd.read_csv('user'+'_'+str(self.target))            
    def _reduce(self):      
        for i in ['browser','os','locale','location']:
                tem = self.pattern[i].unique()
                for j in (self.data[i].unique()):
                    if j not in tem:
                        self.data[i].replace(to_replace = j,value = 'Other',inplace = True)
        for i in range(0,self.lensi):
                tem = self.pattern['site'+str(i)].value_counts()[:10].index
                for j in set(self.data["site"+str(i)]):
                    if j not in tem:
                        self.data["site"+str(i)].replace(to_replace=j,value='Other',inplace=True)
    def __repr__(self):
        return repr(self.data)
    def procces(self):
        self._flatten()
        if "user_id" in self.data.columns:
            self._generate_pattern()
        self._set_pattern()
        self._reduce()
        if "user_id" in self.data.columns:
            self.data.to_csv('train_user'+str(self.target))
        else:
            self.data.to_csv('to_pred_user'+str(self.target))
        
# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser')
    parser.add_argument('dataset_path', type=pathlib.Path,
                        help='dataset path to train model' )
    parser.add_argument('--target', type=int,default=0,
                        help='target id' )
    parser.add_argument('--pre_path', type=pathlib.Path,default=None,
                        help='dataset path to make predictions' )
    
    args = parser.parse_args()
    print(vars(args))
    args = vars(args)
    
    train=Data_prep(args['dataset_path'],args['target'])
    train.procces()
    if args['pre_path']:
        pre_data = Data_prep(args['pre_path'],train.get_target())
        pre_data.procces()
# %%
