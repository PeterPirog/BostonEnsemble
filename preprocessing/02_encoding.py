import numpy as np
from pathlib import Path
from numpy import load, save
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import make_scorer
from xgboost import XGBRegressor
import xgbfir

import joblib
from category_encoders import OneHotEncoder
from feature_engine.selection import DropConstantFeatures, DropDuplicateFeatures, SmartCorrelatedSelection
from Transformers import QuantileTransformerDf, IterativeImputerDf, RareLabelNanEncoder

import pandas as pd

pd.set_option('display.max_columns', None)


def make_preprocessing(config):
    base_path=config['base_path']

    x_train_enc_path=(base_path / "./data_files/encoded_train_X_data.csv").resolve()
    x_train_enc_path_xlsx = (base_path / "./data_files/encoded_train_X_data.xlsx").resolve()
    y_train_path = (base_path / "./data_files/y_train.csv").resolve()

    x_test_enc_path=(base_path / "./data_files/encoded_test_X_data.csv").resolve()
    x_test_enc_path_xlsx = (base_path / "./data_files/encoded_test_X_data.xlsx").resolve()
    y_train_path=(base_path / "./data_files/y_test.csv").resolve()


    #df_x_enc = pd.read_csv(x_train_enc_path)
    #df_y = pd.read_csv(y_train_path)


    #get data from domain encoded files
        #TRAINING DATA
    df_train = pd.read_csv((base_path / "./data_files/domain_train_data.csv").resolve())
    X_train = df_train.drop(['SalePrice','Id'], axis=1).copy()
    Y_train = df_train[['Id','SalePrice']].copy()
    print(f'The X_train shape is:{X_train.shape}')
    #Y_train['SalePrice_1log']=np.log1p(Y_train['SalePrice'])

        # TEST DATA
    df_test= pd.read_csv((base_path / "./data_files/domain_test_data.csv").resolve())


    X_test = df_test.drop(['Id'], axis=1).copy()
    print(f'The X_test shape is:{X_test.shape}')
    #Y_test = df_test[['SalePrice']].copy()

    # print(X_train.head())
    # print(Y_train.head())



    # PREPROCESSING
    # STEP 1 -  categorical features rare labels encoding
    rle = RareLabelNanEncoder(categories=None, tol=config['rare_tol'],
                              minimum_occurrences=None,
                              n_categories=config['n_categories'],
                              max_n_categories=None,
                              replace_with='Rare',
                              impute_missing_label=False,
                              additional_categories_list=None)

    # STEP 2 - categorical features one hot encoding
    # https://github.com/scikit-learn-contrib/category_encoders/blob/master/category_encoders/one_hot.py
    ohe = OneHotEncoder(verbose=0, cols=None, drop_invariant=False, return_df=True,
                        handle_missing='return_nan',
                        # options are 'error', 'return_nan', 'value', and 'indicator'.
                        handle_unknown='return_nan',
                        # options are 'error', 'return_nan', 'value', and 'indicator'
                        use_cat_names=False)

    # STEP 3 - numerical values quantile transformation with skewness removing
    q_trans = QuantileTransformerDf(n_quantiles=1000, output_distribution='uniform', #normal or uniform
                                    ignore_implicit_zeros=False,
                                    subsample=1e5, random_state=42, copy=True, dataframe_as_output=True,
                                    dtype=np.float32)

    dcf = DropConstantFeatures(tol=0.94,
                               missing_values='ignore')
    ddf = DropDuplicateFeatures()

    # STEP 4 - missing values multivariate imputation
    imp = IterativeImputerDf(min_value=-np.inf,  # values from 0 to 1 for categorical for numeric
                             max_value=np.inf,
                             random_state=42,
                             initial_strategy='median',
                             max_iter=config['max_iter'],
                             tol=config['iter_tol'],
                             verbose=3, dataframe_as_output=True)


    scs = SmartCorrelatedSelection(
        variables=None,
        method="pearson",
        threshold=0.9,
        missing_values="ignore",
        selection_method="variance",
        estimator=None,
    )

    # STEP 5 MAKE PIPELINE AND TRAIN IT
    pipeline = Pipeline([
        ('rare_lab', rle),
        ('one_hot', ohe),
        ('q_trans', q_trans),
        ('drop_quasi_const1', dcf),
        ('drop_duplicate1', ddf),
        ('imputer', imp),
       # ('drop_quasi_const2', dcf),
       # ('drop_duplicate2', ddf),
       # ('smart_correlated_sel', scs),
    ])

    # Pipeline training

    X_train_encoded = pipeline.fit_transform(X_train)
    print('X_train_encoded after', X_train_encoded.shape)

    #TEST DATA

    X_test_encoded = pipeline.transform(X_test) #ma być X_test
    print('X_test_encoded after', X_test_encoded.shape)

    # save X_train_encoded array
    # save(file=train_enc_path, arr=X_train_encoded)
    # save(file=y_train_path, arr=Y_train)

    # save trained pipeline
    # joblib.dump(pipeline, pipeline_path)
    df_train_encoded = pd.concat([X_train_encoded, Y_train], axis=1)
    #print(df_train_encoded.head())

    df_train_encoded.to_csv(
        path_or_buf=x_train_enc_path,
        sep=',',
        header=True,
        index=False)
    df_train_encoded.to_excel(
        x_train_enc_path_xlsx,
        sheet_name='output_data',
        index=False)


    X_test_encoded.to_csv(
        path_or_buf=x_test_enc_path,
        sep=',',
        header=True,
        index=False)
    X_test_encoded.to_excel(
        x_test_enc_path_xlsx,
        sheet_name='output_data',
        index=False)





    # STEP 7 SPLITTING DATA FOR KERAS
    X_train, X_test, y_train, y_test = train_test_split(X_train_encoded, Y_train,
                                                        shuffle=True,
                                                        test_size=0.2,
                                                        random_state=42)

    return X_train_encoded, Y_train
    # return X_train, X_test, y_train, y_test


def rmsle(y_true, y_pred, **kwargs):
    # Implementation of rmsle error
    # Convert nparrays  to tensors

    # Clip values to prevent log from values below 0
    y_true = np.clip(y_true, a_min=0, a_max=np.inf)
    y_pred = np.clip(y_pred, a_min=0, a_max=np.inf)
    return -np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


if __name__ == "__main__":
    # define path to the project directory
    base_path = Path(__file__).parent.parent

    # https://machinelearningmastery.com/xgboost-for-regression/
    config = {
        # PREPROCESSING
        # Rare label encoder
        "rare_tol": 0.05,
        "n_categories": 1,
        # Iterative imputer
        "max_iter": 2,
        "iter_tol": 1e-3,
        "output": 'df',
        'base_path': base_path
    }

    import joblib
    from ray.util.joblib import register_ray

    register_ray()
    with joblib.parallel_backend('ray'):
        X, y = make_preprocessing(config=config)

        #print(X.head())
        #print(f'The input shape is:{X.shape}')
