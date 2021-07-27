from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import RepeatedKFold, cross_val_score


class FeatureByNameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, all_features, n_features):
        super().__init__()
        self.all_features = all_features
        self.n_features = n_features
        self.feature_list = self.all_features[:self.n_features]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        return X_[self.feature_list]


class Validator():
    def __init__(self, model_or_pipeline, X, y, n_splits=10, n_repeats=5, random_state=1,scoring='neg_root_mean_squared_error'):
        self.model = model_or_pipeline
        self.X = X
        self.y = y

        # cv parameters
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.scoring = scoring

        #results
        self.M=None
        self.STD=None
        self.UBC=None

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
        return self.M, self.STD, self.UBC
