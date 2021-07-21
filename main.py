import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer as _IterativeImputer
from sklearn.preprocessing import QuantileTransformer as _QuantileTransformer

#https://github.com/scikit-learn/scikit-learn/issues/5523
def check_output(X, ensure_index=None, ensure_columns=None):
    """
    Joins X with ensure_index's index or ensure_columns's columns when avaialble
    """
    if ensure_index is not None:
        if ensure_columns is not None:
            if type(ensure_index) is pd.DataFrame and type(ensure_columns) is pd.DataFrame:
                X = pd.DataFrame(X, index=ensure_index.index, columns=ensure_columns.columns)
        else:
            if type(ensure_index) is pd.DataFrame:
                X = pd.DataFrame(X, index=ensure_index.index)
    return X

class IterativeImputer(_IterativeImputer):
    def transform(self, X):
        Xt = super(IterativeImputer, self).transform(X)
        return check_output(Xt, ensure_index=X, ensure_columns=X)
    def fit_transform(self, X,y):
        Xt = super(IterativeImputer, self).fit_transform(X)
        return check_output(Xt, ensure_index=X, ensure_columns=X)

class QuantileTransformer(_QuantileTransformer):
    def transform(self, X):
        Xt = super(QuantileTransformer, self).transform(X)
        return check_output(Xt, ensure_index=X, ensure_columns=X)
    def fit_transform(self, X,y):
        Xt = super(QuantileTransformer, self).fit_transform(X)
        return check_output(Xt, ensure_index=X, ensure_columns=X)

if __name__ == "__main__":
    X_train = pd.DataFrame([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]], columns=['aaa', 'bbb', 'ccc'])
    X_test = pd.DataFrame([[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9], [10, 4.5, np.nan]],
                          columns=['aaa', 'bbb', 'ccc'])
    print(X_train.head())

    #imp_mean = IterativeImputerDf(random_state=0, dataframe_as_output=True)


    imp_mean = IterativeImputer(random_state=0)
    quantile=QuantileTransformer()

    pipeline = Pipeline([
        ('rare_lab', quantile),
        ('one_hot', imp_mean ),
    ])



    #imp_mean.fit(X=X_train)
    out_train = pipeline.fit_transform(X=X_train)
    out_test = pipeline.transform(X=X_test)

    print('input_train=\n', out_train)
    print('out_train=\n', out_train)
    print('out_test=\n', out_test)
