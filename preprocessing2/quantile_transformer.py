import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin


class CategoriclalQuantileEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, features=None, p=0.5, m=1, remove_original=True):
        super().__init__()
        self.features = features  # selected categorical features
        self.columns = None  # all columns in df
        self.p=p
        self.m=m

        if isinstance(p,int) or isinstance(p,float):
            self.p =[self.p]
        if isinstance(m,int) or isinstance(m,float):
            self.m =[self.m]

        print(self.m)
        print(self.p)

        self.remove_original = remove_original

    def fit(self, X, y=None):

        self.columns = X.columns
        # Find only categorical columns if not defines
        if self.features is None:
            self.features = [col for col in self.columns if X[col].dtypes == 'O']
        else:
            if isinstance(self.features,str): #convert single feature name to list for iteration possibility
                self.features=[self.features]

        print(self.columns)
        print(self.features)
        return self

    def transform(self, X):
        X = X.copy()

        return X


if __name__ == '__main__':
    df_train = pd.read_csv('/home/peterpirog/PycharmProjects/BostonEnsemble/data_files/train.csv')
    df_test = pd.read_csv('/home/peterpirog/PycharmProjects/BostonEnsemble/data_files/train.csv')

    features = ['BldgType', 'GrLivArea', 'MSSubClass', 'OverallQual']
    X = df_train[features].copy()

    y = np.log1p(df_train['SalePrice']).copy()

    X['MSSubClass'] = 'm' + X['MSSubClass'].apply(str)
    print(df_train.head(5))
    print(y)

    cqe = CategoriclalQuantileEncoder(features='MSSubClass',p=[0.1,0.5,0.9],m=[1,10])

    X_enc = cqe.fit_transform(X=X, y=y)
    print(X_enc.head())
