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
        self.column_target = None
        self.p = p
        self.m = m
        self.features_unique = {}  # dict with unique values for speciffied feature
        #self.qp = []  # quantiles for all dataset
        self.global_quantiles={} #stored quantiles for whole dataset
        self.value_quantiles = {} #stored quantiles for all values, key form (feature, value, p)
        self.value_counts={} #stored counts of every value in train data key form (feature, value)

        # convert p and m to lists for iteration
        if isinstance(p, int) or isinstance(p, float):
            self.p = [self.p]
        if isinstance(m, int) or isinstance(m, float):
            self.m = [self.m]

        # print(self.m)

        self.remove_original = remove_original

    def fit(self, X, y=None):
        y = y.to_frame().copy()
        Xy = pd.concat([X, y], axis=1)
        print(Xy)
        self.columns = X.columns
        # Find only categorical columns if not defines
        if self.features is None:
            self.features = [col for col in self.columns if X[col].dtypes == 'O']
        else:
            if isinstance(self.features, str):  # convert single feature name to list for iteration possibility
                self.features = [self.features]

        # Find unique values for specified features
        for feature in self.features:
            self.features_unique[feature] = list(X[feature].unique())

        # Find quantiles for all dataset for each value of p
        for p in self.p:
            #self.qp.append(np.quantile(y, p))
            self.global_quantiles[p]=np.quantile(y, p)

        #print(f'global quantiles={self.global_quantiles}')
        # Find quantiles for evry faeature and every value
        for feature in self.features:
            for value in list(X[feature].unique()):  # for every unique value
                # Find y values for specified feature and specified value
                idx = Xy[feature] == value
                y_group = y[idx]

                self.value_counts[feature, value] = len(y_group) #counts for every feature and every value
                print(f'n={self.value_counts[feature, value]} for feature={feature}, value={value}')

                for p in self.p:
                    self.value_quantiles[feature, value, p] = np.quantile(y_group, p)

        print(f'value_quantiles={self.value_quantiles}')
        print(f'value_counts={self.value_counts}')

        # print(self.columns)
        # print(self.features)
        return self

    def transform(self, X):
        X = X.copy()

        # Create new columns for quantile values
        for feature in self.features:
            for p in self.p:
                for m in self.m:
                    feature_name = feature + '_' + str(p) + '_' + str(m)
                    X[feature_name] = np.nan
        # print(feature_name)

        return X


if __name__ == '__main__':
    df_train = pd.read_csv('/home/peterpirog/PycharmProjects/BostonEnsemble/data_files/train.csv')
    df_test = pd.read_csv('/home/peterpirog/PycharmProjects/BostonEnsemble/data_files/train.csv')

    features = ['BldgType', 'GrLivArea', 'MSSubClass', 'OverallQual']
    X = df_train[features].copy()

    y = np.log1p(df_train['SalePrice']).copy()

    X['MSSubClass'] = 'm' + X['MSSubClass'].apply(str)
    # print(df_train.head(5))
    # print(y)

    cqe = CategoriclalQuantileEncoder(features=None, p=[0.1, 0.5, 0.9], m=[1, 10])

    X_enc = cqe.fit_transform(X=X, y=y)
    print(X_enc.head())
