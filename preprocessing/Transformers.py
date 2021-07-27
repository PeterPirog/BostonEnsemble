#https://github.com/scikit-learn-contrib/category_encoders/blob/master/category_encoders/one_hot.py
#pip install category_encoders
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer as _QuantileTransformer
from feature_engine.encoding import RareLabelEncoder

from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer as _IterativeImputer

from category_encoders import one_hot

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

class RareLabelNanEncoder(BaseEstimator, TransformerMixin):
    """This function based on:
    https://feature-engine.readthedocs.io/en/latest/encoding/RareLabelEncoder.html
    Additionally makes possible rare label encoding even with missing values,
    if impute_missing_label=False missing values in output dataframe is np.nan
    if impute_missing_label=True missing values in output dataframe is 'MISSING
    """

    def __init__(self, categories=None, tol=0.05, minimum_occurrences=None, n_categories=10, max_n_categories=None,
                 replace_with='Rare', impute_missing_label=False, additional_categories_list=None):
        """
        :param categories:
        :param tol: The minimum frequency a label should have to be considered frequent. Categories with frequencies lower than tol will be grouped
        :param minimum_occurrences: defined minimum number of value occurrences for single feature
        :param n_categories: The minimum number of categories a variable should have for the encoder to find frequent labels. If the variable contains less categories, all of them will be considered frequent.
        :param max_n_categories: The maximum number of categories that should be considered frequent. If None, all categories with frequency above the tolerance (tol) will be considered frequent.
        :param replace_with: The category name that will be used to replace infrequent categories.
        :param impute_missing_label: if  False missing values in output dataframe is np.nan if True missing values in output dataframe is 'MISSING
        :param additional_categories_list: add list with feature if you want  feature for default founded categorical features
        """
        super().__init__()
        self.categories = categories
        self.additional_categories_list = additional_categories_list
        self.impute_missing_label = impute_missing_label
        self.new_categories = []
        self.number_of_samples = None
        self.minimum_occurrences = minimum_occurrences

        # original RareLabelEncoder parameters
        self.tol = tol
        self.n_categories = n_categories
        self.max_n_categories = max_n_categories
        self.replace_with = replace_with

    def fit(self, X, y=None):
        X = X.copy()
        self.number_of_samples = X.shape[0]  # number of rows in dataframe

        if self.categories is None:
            self.categories = X.select_dtypes(include=['object']).columns.tolist()
            # option to add some additional feature if you need
            if self.additional_categories_list is not None:
                self.categories = self.categories + self.additional_categories_list

        # option to define minimum value occurrence for single feature- usefull for huge datasets with high cardinality
        if self.minimum_occurrences is not None:
            self.tol = float(self.minimum_occurrences / self.number_of_samples)
            print(f'Value of minimum_occurrences is defined. New value of tol is:{self.tol}')

        return self

    def transform(self, X, y=None):
        pd.options.mode.chained_assignment = None  # default='warn' - turn off warning about  data overwrite
        for category in self.categories:
            x = X[category].copy()  # not use copy to intentionally change value
            idx_nan = x.loc[pd.isnull(x)].index  # find nan values in analyzed feature column

            # replace missing values
            x[idx_nan] = 'MISS'
            encoder = RareLabelEncoder(tol=self.tol, n_categories=self.n_categories,
                                       max_n_categories=self.max_n_categories,
                                       replace_with=self.replace_with)

            x = x.to_frame(name=category)  # convert pd.series to dataframe
            x = encoder.fit_transform(x)
            X[category] = x
            if not self.impute_missing_label:
                X[category].loc[idx_nan] = np.nan
        pd.options.mode.chained_assignment = 'warn'  # default='warn' - turn on warning about  data overwrite
        return X


class OneHotNanEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categories='auto', drop=True, dtype=np.float64):
        super().__init__()
        self.categories = categories
        self.drop = drop
        self.dtype = dtype
        self.new_categories = []

    def fit(self, X, y=None):

        if self.categories == 'auto':
            self.categories = X.select_dtypes(include=['object']).columns.tolist()

        # get new categories based on train data
        for category in self.categories:
            labels = X[category].unique().tolist()
            labels = [str(x) for x in labels]  # converting nan to 'nan'
            if 'nan' in labels:
                labels.remove('nan')  # remove nan labels

            for label in labels:
                new_label = str(category) + '_' + str(label)
                self.new_categories.append(new_label)

        self.new_categories=list(set(self.new_categories)) #get unique elements
        print(f'new_categories={self.new_categories}')

        return self

    def transform(self, X, y=None):
        X = X.copy()
        X[self.new_categories]=np.nan
        """
        for category in self.categories:
            labels = X[category].unique().tolist()
            labels = [str(x) for x in labels]  # converting nan to 'nan
            if 'nan' in labels:
                labels.remove('nan')  # remove nan labels

            for label in labels:
                new_label = str(category) + '_' + str(label)
                self.new_categories.append(new_label)
                X[new_label] = np.where(X[category] == label, 1, 0)
                X.loc[X[category].isna(), new_label] = np.nan
        if self.drop:
            X = X.drop(columns=self.categories)  # drop encoded columns
        # X[self.new_categories] = X[self.new_categories].astype(self.dtype)
        """
        return X
