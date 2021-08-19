import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin


class CategoriclalQuantileEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, features=None, p=0.5, m=1, remove_original=True, return_df=True, handle_missing='value',
                 handle_unknown='value'):
        super().__init__()
        self.features = features  # selected categorical features
        self.columns = None  # all columns in df
        self.column_target = None
        self.p = p
        self.m = m
        self.return_df = return_df
        self.handle_missing = handle_missing  # 'value' or ‘return_nan’
        self.handle_unknown = handle_unknown  # 'value' or ‘return_nan’

        self.features_unique = {}  # dict with unique values for speciffied feature
        # self.qp = []  # quantiles for all dataset
        self.global_quantiles = {}  # stored quantiles for whole dataset, key form (p)
        self.value_quantiles = {}  # stored quantiles for all values, key form (feature, value, p)
        self.value_counts = {}  # stored counts of every value in train data key form (feature, value)

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
            # self.qp.append(np.quantile(y, p))
            self.global_quantiles[p] = np.quantile(y, p)

        # print(f'global quantiles={self.global_quantiles}')
        # Find quantiles for evry faeature and every value
        for feature in self.features:
            for value in list(X[feature].unique()):  # for every unique value
                # Find y values for specified feature and specified value
                idx = Xy[feature] == value
                y_group = y[idx]

                self.value_counts[feature, value] = len(y_group)  # counts for every feature and every value
                print(f'n={self.value_counts[feature, value]} for feature={feature}, value={value}')

                for p in self.p:
                    self.value_quantiles[feature, value, p] = np.quantile(y_group, p)
        print(f'global_quantiles={self.global_quantiles}')
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
                    # Quantile Encoder: Tackling High Cardinality Categorical Features in Regression Problems, equation 2

                    if self.handle_missing == 'value':  # return in output df global quantile values if input value is nan
                        X[feature_name]=self.global_quantiles[p]
                        X[feature_name] = X[feature].apply(lambda value: (self.value_counts[feature, value] *
                                                                          self.value_quantiles[feature, value, p] + m *
                                                                          self.global_quantiles[p])
                                                                         / (self.value_counts[
                                                                                feature, value] + m) if not pd.isnull(
                            value) else self.global_quantiles[p])

                    if self.handle_missing == 'return_nan':  # return in output df np.nan if input value is nan
                        X[feature_name] = X[feature].apply(lambda value: (self.value_counts[feature, value] *
                                                                          self.value_quantiles[feature, value, p] + m *
                                                                          self.global_quantiles[p])
                                                                         / (self.value_counts[
                                                                                feature, value] + m) if not pd.isnull(
                            value) else np.nan)

        # Remove original features
        if self.remove_original:
            X = X.drop(self.features, axis=1)

        # Return dataframe or np array
        if self.return_df:
            return X
        else:
            return X.to_numpy()


if __name__ == '__main__':
    df_train = pd.read_csv('/home/peterpirog/PycharmProjects/BostonEnsemble/data_files/train.csv')
    df_test = pd.read_csv('/home/peterpirog/PycharmProjects/BostonEnsemble/data_files/test_nan_unknown.csv')

    features = ['BldgType', 'GrLivArea', 'MSSubClass', 'OverallQual']
    X = df_train[features].copy()

    y = np.log1p(df_train['SalePrice']).copy()

    X['MSSubClass'] = 'm' + X['MSSubClass'].apply(str)
    # print(df_train.head(5))
    # print(y)

    cqe = CategoriclalQuantileEncoder(features=None, p=[0.1, 0.5, 0.9], m=[0, 50],
                                      remove_original=True,
                                      return_df=True,
                                      handle_missing='value',
                                      handle_unknown='value')

    X_enc = cqe.fit_transform(X=X, y=y)
    print(X_enc.head())
    X_test=df_test[features].copy()
    X_test['MSSubClass'] = 'm' + X_test['MSSubClass'].apply(str)

    print(X_test.head())
    #out=cqe.transform(X=X_test)