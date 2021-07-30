from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import RepeatedKFold, cross_val_score
import json
import numpy as np
import pandas as pd
import dill as pickle


class FeatureByNameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, all_features, n_features=None):
        super().__init__()
        self.all_features = all_features
        if n_features is None:
            self.n_features = len(self.all_features)
        else:
            self.n_features = n_features
        self.feature_list = self.all_features[:self.n_features]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        return X_[self.feature_list]

    def predict(self, X):
        X_ = X.copy()
        return X_[self.feature_list]


class Validator():
    def __init__(self, model_or_pipeline, X, y, n_splits=10, n_repeats=5, random_state=1,
                 scoring='neg_root_mean_squared_error', model_config_dict=None):
        self.model = model_or_pipeline
        self.X = X
        self.y = y
        self.config_dict = model_config_dict

        # cv parameters
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.scoring = scoring

        # results
        self.M = None
        self.STD = None
        self.UBC = None

        self.cv = RepeatedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_state)
        # evaluate_model

    def run(self):
        self.scores = cross_val_score(self.model, self.X, self.y,
                                      scoring=self.scoring,
                                      cv=self.cv,
                                      n_jobs=-1)

        # force scores to be positive
        self.scores = abs(self.scores)
        self.M = self.scores.mean()
        self.STD = self.scores.std()
        self.UBC = self.M + 2 * self.STD

        print('Mean error: %.4f (%.4f), UBC= %.4f' % (self.M, self.STD, self.UBC))

        if self.config_dict is None:
            pass
        else:
            self.config_dict['M'] = self.M
            self.config_dict['STD'] = self.STD
            self.config_dict['UBC'] = self.UBC
            with open(self.config_dict['json_file'], 'w') as fp:
                json.dump(self.config_dict, fp)
        return self.M, self.STD, self.UBC


def create_submission_from_nparray(predicted_array,
                                   test_csv_path='/home/peterpirog/PycharmProjects/BostonEnsemble/data_files/test.csv'):
    # convert logarithm value to original scale
    y_pred = np.exp(predicted_array) - 1

    # Get indexes from test file
    df_idx = pd.read_csv(test_csv_path)
    idx = df_idx['Id'].to_numpy().astype(np.int)

    # stack Id and price column
    out_arr = np.column_stack((idx, y_pred))
    labels = ['Id', 'SalePrice']

    df = pd.DataFrame(data=out_arr, columns=labels)
    df['Id'] = df['Id'].astype('int32')

    df.to_csv(path_or_buf='/home/peterpirog/PycharmProjects/BostonEnsemble/data_files/my_submission.csv',
              sep=',', index=False)
    df.to_excel('/home/peterpirog/PycharmProjects/BostonEnsemble/data_files/my_submission.xlsx',
                sheet_name='output_data',
                index=False)

def model_to_submission(model_file_pkl,X):

    with open(model_file_pkl, 'rb') as file:
        model = pickle.load(file)
        y_predicted=model.predict(X)
        create_submission_from_nparray(y_predicted)