"""[summary]
Python module to prepare data before fitting into model
Supports command line arguments
dataset_path, type:pathlib.Path dataset path to train model
--target, type:int,default=0, target id
--pre_path, type:pathlib.Path, default=None, dataset path to make predictions
Returns:
    Pandas DataFrame saved to csv file
"""
import argparse
import pathlib
import pandas as pd
import numpy as np
class DataPrep:
    """[summary]
    Class object containing data engineering functions
    Data is loaded form .json file
    Transformed into flat table
    Pattern is set to rows that contain given target
    Rest of data is transformed using pattern
    If pattern cannot be made it uses one made previosly to given target
    Saves changed data and pattern to csv files
    Returns:
        Pandas DataFrame : Dataframe with changed values according to target pattern
    """
    def __init__(self,path,target_id):
        """[summary]
        Object initialization
        Args:
            path (pathlib.Path): .json file path from with data will be loaded
            target_id (Int): Integer value, id of user that pattern will be based of
        """
        self.path = path
        self.data = pd.read_json(self.path)
        self.target = target_id
        self.lensi = 0
        self.pattern = None
    def get_lensi(self):
        """[summary]
        Getter method of lensi value,
        Max length of sites dictonary in .json file
        Returns:
            Int: Max number of sites
        """
        return self.lensi
    def set_lensi(self,length):
        """[summary]
        Setter method of lensi value
        Assing given lenght value to object attribute lensi
        Args:
            length (Int): Length on dictonary in .json file
        """
        self.lensi = length
    def get_target(self):
        """[summary]
        Getter method of target value
        Return object target attribute
        Returns:
            Int: Id number
        """
        return self.target
    def set_target(self,target_id):
        """[summary]
        Setter method target attribute
        Args:
            target_id (Int): Sets given number to attribute target
        """
        self.target = target_id
    def get_pattern(self):
        """[summary]
        Getter method pattern attribute
        Returns:
            Pandas DataFrame: DataFrame only with rows that contains user_id as in target attribute
        """
        return repr(self.pattern)
    def _generate_pattern(self):
        """[summary]
        Method
        Chooses rows from data attribute that has only user_id as in target attribute.
        Saves it into csv file
        """
        separeted = self.data[self.data['user_id']==self.target]
        separeted.to_csv('user'+'_'+str(self.target)+".csv",index = False)
    def _set_pattern(self):
        """[summary]
        Setter method
        Loads csv file matching target attribute
        """
        self.pattern = pd.read_csv('user'+'_'+str(self.target)+".csv")
    def _flatten(self):
        """[summary]
        Method
        Flattens .json file
        """
        sites = pd.json_normalize(self.data.sites)
        self.set_lensi(len(sites.columns))
        temp_frame=self.data.drop('sites',axis = 1)
        for i in range(0,len(sites.columns)):
            temp_frame['site'+str(i)] = pd.json_normalize(sites[i]).site
            temp_frame['site_len'+str(i)] = pd.json_normalize(sites[i]).length
            temp_frame['site'+str(i)].fillna('None',inplace=True)
            temp_frame['site_len'+str(i)].fillna(0,inplace = True)
        self.data = temp_frame
        self.data["time"] = pd.to_timedelta(self.data["time"])
        self.data["time"] = (self.data["time"].dt.components["hours"]*60
                            + self.data["time"].dt.components["minutes"])
        self.data['date'] = pd.to_datetime(self.data['date']).view(np.int64)
        self.data["gender"].replace({'m':0,'f':1},inplace = True)
    def _reduce(self):
        """[summary]
        Method
        Reduces cardinality of data using pattern attribute
        """
        for i in ['browser','os','locale','location']:
            tem = self.pattern[i].unique()
            for j in self.data[i].unique():
                if j not in tem:
                    self.data[i].replace(to_replace = j,value = 'Other',inplace = True)
        for i in range(0,self.lensi):
            tem = self.pattern['site'+str(i)].value_counts()[:10].index
            for j in set(self.data["site"+str(i)]):
                if j not in tem:
                    self.data["site"+str(i)].replace(to_replace=j,value='Other',inplace=True)
    def __repr__(self):
        """[summary]
        Method
        Returns object data as Pandas DataFrame
        Returns:
            Pandas DataFrame: Object data attribute
        """
        return repr(pd.DataFrame(self.data))
    def procces(self):
        """[summary]
        Has 3 or 4 steps
        #1 Uses flatten method
        ##2 If there is user_is column in flatten data it generates pattern, generate_pattern method
        #3 Loads pattern to given target, set_pattern method
        #4 Uses pattern to reduce cardinality, reduce method
        Saves proccesed data into csv file if there was step 2 then it is train data file
        """
        self._flatten()
        if "user_id" in self.data.columns:
            self._generate_pattern()
        self._set_pattern()
        self._reduce()
        if "user_id" in self.data.columns:
            self.data.to_csv('train_user'+str(self.target)+".csv",index = False)
        else:
            self.data.to_csv('to_pred_user'+str(self.target)+".csv",index = False)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser')
    parser.add_argument('dataset_path', type=pathlib.Path,
                        help='dataset path to train model')
    parser.add_argument('--target', type=int,default=0,
                        help='target id')
    parser.add_argument('--pre_path', type=pathlib.Path,default=None,
                        help='dataset path to make predictions')
    args = vars(parser.parse_args())
    print(args)
    train=DataPrep(args['dataset_path'],args['target'])
    train.procces()
    if args['pre_path']:
        pre_data = DataPrep(args['pre_path'],train.get_target())
        pre_data.procces()
        