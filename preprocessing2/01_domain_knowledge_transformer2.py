#%%
from pathlib import Path

import numpy as np
import pandas as pd
from feature_engine.selection import DropConstantFeatures, SmartCorrelatedSelection
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline

from transformer_tools import CategoriclalQuantileEncoder, \
    DomainKnowledgeTransformer, \
    QuantileTransformer, IterativeImputer
from pyearth import Earth
from sklearn.neighbors import KNeighborsRegressor

if __name__ == '__main__':
    verbose = False

    # define path to the project directory
    project_path = Path(__file__).parent.parent

    # make all dataframe columns visible
    pd.set_option('display.max_columns', None)

    #%%
    df_train = pd.read_csv(project_path.joinpath('data_files/train.csv'))
    # %%
    df_test = pd.read_csv(project_path.joinpath('data_files/test.csv'))

    X = df_train.drop(['SalePrice'], axis='columns').copy()
    y = np.log1p(df_train['SalePrice']).copy()

    X_test = df_test.copy()

    dkt = DomainKnowledgeTransformer()
    cqe = CategoriclalQuantileEncoder(features=None,
                                      ignored_features=None,
                                      p=[0.1, 0.5, 0.9], m=[1,10,50],
                                      remove_original=True,
                                      return_df=True,
                                      handle_missing_or_unknown='value')
    dcf = DropConstantFeatures(tol=0.99, missing_values='ignore')
    q_trans = QuantileTransformer(n_quantiles=1000, output_distribution='normal',  # normal or uniform
                                  ignore_implicit_zeros=False,
                                  subsample=1e5, random_state=42, copy=True)

    scs = SmartCorrelatedSelection(variables=None,
                                   method="pearson",
                                   threshold=0.9,
                                   missing_values="ignore",  # ignore or raise
                                   selection_method="variance",
                                   estimator=None,
                                   )

    q_trans = QuantileTransformer(n_quantiles=1000, output_distribution='normal',  # normal or uniform
                                  ignore_implicit_zeros=False,
                                  subsample=1e5, random_state=42, copy=True)

    imp = IterativeImputer(min_value=-np.inf,  # values from 0 to 1 for categorical for numeric
                           max_value=np.inf,
                           random_state=42,
                           initial_strategy='median',
                           max_iter=10,
                           tol=0.001,
                           verbose=3)


    # X_enc = dkt.fit_transform(X=X, y=y)
    # print(X_enc.head(5))

    # X_enc = cqe.fit_transform(X=X_enc, y=y)
    # print(X_enc.head(5))

    pipe = Pipeline([('domain_transformer', dkt),
                     ('categorical_encoder', cqe),
                     ('drop_quasi_constant', dcf),
                     ('quantile_scaling',q_trans),
                     #('smart_correlation',scs),
                     ('imputer',imp)])

    X_encoded = pipe.fit_transform(X=X, y=y)
    #y_out = pipe.transform(X=X_test)

    #print(y_out.head(10))
    print(X_encoded.info())
    # print(y_out .describe())
    n_features=len(X_encoded.columns)
    print(f'n_features={n_features}')


    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)
    sfs = SFS(Earth(),
              k_features=100,
              forward=True,
              floating=True,
              verbose=2,
              scoring='neg_root_mean_squared_error',  # 'neg_root_mean_squared_error' 'r2'
              cv=cv,
              n_jobs=-1)

    sfs = sfs.fit(X=X_encoded, y=y)
    #print(X.columns[list(sfs.k_feature_idx_)])

    df = pd.DataFrame.from_dict(sfs.get_metric_dict(confidence_interval=0.95)).T
    df.to_csv('forward_features_Earth_100.csv')
    df.to_excel('forward_features_Earth_100.xlsx')
"""
Elastic Net 50 features -0,121143288
['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtQual', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtUnfSF', 'TotalBsmtSF', 'HeatingQC', 'CentralAir', 'GrLivArea', 'BsmtFullBath', 'FullBath', 'HalfBath', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageCars', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch', 'MSZoning_0.5_10', 'MSZoning_0.9_1', 'MSZoning_0.9_10', 'LandContour_0.5_1', 'LotConfig_0.9_1', 'Neighborhood_0.1_10', 'Neighborhood_0.5_1', 'Neighborhood_0.9_1', 'Neighborhood_0.9_10', 'Condition1_0.5_1', 'Condition1_0.5_10', 'Condition1_0.9_1', 'BldgType_0.9_10', 'HouseStyle_0.9_10', 'RoofStyle_0.5_10', 'RoofMatl_0.9_1', 'Exterior1st_0.5_1', 'Foundation_0.1_10', 'Foundation_0.5_10', 'GarageType_0.1_10', 'MiscFeature_0.5_1', 'MoSold_0.9_1', 'SaleCondition_0.1_10', 'SaleCondition_0.5_1', 'SaleCondition_0.5_10']


"""