import numpy as np
import pandas as pd
from feature_engine.encoding import RareLabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
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

class CategoriclalQuantileEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, features=None, ignored_features=None, p=0.5, m=1, remove_original=True, return_df=True,
                 handle_missing_or_unknown='value'):
        super().__init__()
        self.features = features  # selected categorical features
        self.ignored_features = ignored_features
        self.columns = None  # all columns in df
        self.column_target = None
        self.p = p
        self.m = m
        self.remove_original = remove_original
        self.return_df = return_df
        self.handle_missing_or_unknown = handle_missing_or_unknown  # 'value' or ‘return_nan’

        self.features_unique = {}  # dict with unique values lists for specified feature, key form (feature)
        self.global_quantiles = {}  # stored quantiles for whole dataset, key form (p)
        self.value_quantiles = {}  # stored quantiles for all values, key form (feature, value, p)
        self.value_counts = {}  # stored counts of every value in train data key form (feature, value)

        # convert p and m to lists for iteration available
        if isinstance(p, int) or isinstance(p, float):
            self.p = [self.p]
        if isinstance(m, int) or isinstance(m, float):
            self.m = [self.m]

        # convert feature lists for iteration available
        if not isinstance(self.features, list) and self.features is not None:
            self.features = [self.features]

        if not isinstance(self.ignored_features, list) and self.ignored_features is not None:
            self.ignored_features = [self.ignored_features]

    def fit(self, X, y=None):
        y = y.to_frame().copy()
        Xy = pd.concat([X, y], axis=1)
        self.columns = X.columns
        # Find only categorical columns if not defines
        if self.features is None:
            self.features = [col for col in self.columns if X[col].dtypes == 'O']
        else:
            if isinstance(self.features, str):  # convert single feature name to list for iteration possibility
                self.features = [self.features]
        # Remove ignored features
        if self.ignored_features is not None:
            for ignored_feature in self.ignored_features:
                self.features.remove(ignored_feature)

        # Find unique values for specified features
        for feature in self.features:
            self.features_unique[feature] = list(X[feature].unique())

        # Find quantiles for all dataset for each value of p
        for p in self.p:
            self.global_quantiles[p] = np.quantile(y, p)

        # Find quantiles for every feature and every value
        for feature in self.features:
            for value in list(X[feature].unique()):  # for every unique value
                # Find y values for specified feature and specified value
                idx = Xy[feature] == value
                y_group = y[idx]
                # counts for every feature and every value
                self.value_counts[feature, value] = len(y_group)
                #print(f'n={self.value_counts[feature, value]} for feature={feature}, value={value}')

                for p in self.p:
                    self.value_quantiles[feature, value, p] = np.quantile(y_group, p)
        return self

    def transform(self, X):
        X = X.copy()

        # Create new columns for quantile values
        for feature in self.features:
            X[feature] = X[feature].replace(np.nan, 'MISSING')
            X[feature] = X[feature].apply(lambda value: value if value in self.features_unique[feature] else 'UNKNOWN')
            for p in self.p:
                for m in self.m:
                    # Prepare new columns names
                    feature_name = feature + '_' + str(p) + '_' + str(m)

                    # return global quantile values if input value is nan or unknown
                    if self.handle_missing_or_unknown == 'value':
                        X[feature_name] = self.global_quantiles[p]
                        X[feature_name] = X[feature].apply(lambda value: self.global_quantiles[p]
                        if value == "MISSING" or value == 'UNKNOWN'
                        # Quantile Encoder: Tackling High Cardinality Categorical Features in Regression Problems, equation 2
                        else (self.value_counts[feature, value] * self.value_quantiles[feature, value, p] +
                              m * self.global_quantiles[p]) / (self.value_counts[feature, value] + m))


                    # return nan if input value is nan or unknown
                    if self.handle_missing_or_unknown == 'return_nan':
                        X[feature_name] = self.global_quantiles[p]
                        X[feature_name] = X[feature].apply(lambda value: np.nan
                        if value == "MISSING" or value == 'UNKNOWN'
                        # Quantile Encoder: Tackling High Cardinality Categorical Features in Regression Problems, equation 2
                        else (self.value_counts[feature, value] * self.value_quantiles[feature, value, p] +
                              m * self.global_quantiles[p]) / (self.value_counts[feature, value] + m))
                    X[feature_name] = X[feature_name].copy()  # defragment
        # Remove original features
        if self.remove_original:
            X = X.drop(self.features, axis=1)

        # Return dataframe or np array
        if self.return_df:
            return X
        else:
            return X.to_numpy()

class DomainKnowledgeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, remove_original=True):
        super().__init__()
        self.remove_original = remove_original

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Remove index column
        try:
            X = X.drop(['Id'], axis=1)
        except:
            pass

        # "MSSubClass" - convert values to objects
        X['MSSubClass'] = 'm' + X['MSSubClass'].apply(str)

        # "MSZoning" - do nothing

        # "LotFrontage" - feature prepare for missing values
        X['LotFrontage'] = np.where(X['LotFrontage'].isnull(), np.nan, X['LotFrontage'].values)

        # "LotArea" - feature prepare for missing values
        X['LotArea'] = np.where(X['LotArea'].isnull(), np.nan, X['LotArea'].values)

        # "Street" - do nothing

        # "Alley" - default value Paved =2, Gravel=1, None=0
        X['Alley'] = X['Alley'].apply(lambda value: 2 if value == 'Pave' else value)
        X['Alley'] = X['Alley'].apply(lambda value: 1 if value == 'Gravel' else value)
        X['Alley'] = X['Alley'].apply(lambda value: 0 if pd.isnull(value) else value)

        # "LotShape" -  Reg =3, IR1=2, IR2=1, IR3=0
        X['LotShape'] = X['LotShape'].apply(lambda value: np.nan if pd.isnull(value) else value)
        X['LotShape'] = X['LotShape'].apply(lambda value: 3 if value == 'Reg' else value)
        X['LotShape'] = X['LotShape'].apply(lambda value: 2 if value == 'IR1' else value)
        X['LotShape'] = X['LotShape'].apply(lambda value: 1 if value == 'IR2' else value)
        X['LotShape'] = X['LotShape'].apply(lambda value: 0 if value == 'IR3' else value)

        # "LandContour" - do nothing

        # "Utilities" - AllPub=3, NoSewr=2, NoSeWa=1,ELO
        X['Utilities'] = X['Utilities'].apply(lambda value: np.nan if pd.isnull(value) else value)
        X['Utilities'] = X['Utilities'].apply(lambda value: 3 if value == 'AllPub' else value)
        X['Utilities'] = X['Utilities'].apply(lambda value: 2 if value == 'NoSewr' else value)
        X['Utilities'] = X['Utilities'].apply(lambda value: 1 if value == 'NoSeWa' else value)
        X['Utilities'] = X['Utilities'].apply(lambda value: 0 if value == 'ELO' else value)

        # "LotConfig" - do nothing

        # "LandSlope" - Gtl=2,Mod=1,Sev=0
        X['LandSlope'] = X['LandSlope'].apply(lambda value: np.nan if pd.isnull(value) else value)
        X['LandSlope'] = X['LandSlope'].apply(lambda value: 2 if value == 'Gtl' else value)
        X['LandSlope'] = X['LandSlope'].apply(lambda value: 1 if value == 'Mod' else value)
        X['LandSlope'] = X['LandSlope'].apply(lambda value: 0 if value == 'Sev' else value)

        # "Neighborhood" - do nothing
        # Condition 1 and Condition 2 - do nothing - UPGRADE POSSIBLE
        # "BldgType" -do nothing
        # "HouseStyle" -do nothing
        # "OverallQual" -do nothing
        # "OverallCond" -do nothing

        # "YearBuilt YearRemodAdd" - default value np.nan
        X['BuildingAge'] = X['YrSold'] - X['YearBuilt']
        X['YearsFromRenovation'] = X['YrSold'] - X['YearRemodAdd']

        # "RoofStyle" - do nothing
        # "RoofMatl" - do nothing

        #  Exterior1st and Exterior2nd - do nothing - UPGRADE POSSIBLE
        # "MasVnrType"- do nothing
        X['MasVnrType'] = X['MasVnrType'].apply(lambda value: 'Miss' if pd.isnull(value) else value)

        # "ExterQual" - Ex=4, Gd=3, TA=2, Fa=1, Po=0
        X['ExterQual'] = X['ExterQual'].apply(lambda value: self.__convert_str_to_int(value))

        # "ExterCond" - Ex=4, Gd=3, TA=2, Fa=1, Po=0
        X['ExterCond'] = X['ExterCond'].apply(lambda value: self.__convert_str_to_int(value))

        # "Foundation" - do nothing

        # "BsmtQual" - Ex=4, Gd=3, TA=2, Fa=1, Po=0, NA=-1
        X['BsmtQual'] = X['BsmtQual'].apply(lambda value: self.__convert_str_to_int(value))
        X['BsmtQual'] = X['BsmtQual'].apply(lambda value: -1 if pd.isnull(value) else value)

        # "BsmtCond" - Ex=4, Gd=3, TA=2, Fa=1, Po=0, NA=-1
        X['BsmtCond'] = X['BsmtCond'].apply(lambda value: self.__convert_str_to_int(value))
        X['BsmtCond'] = X['BsmtCond'].apply(lambda value: -1 if pd.isnull(value) else value)

        # "BsmtExposure" - Gd=4, Av=3, Mn=2, No=1, NA=0
        X['BsmtExposure'] = X['BsmtExposure'].apply(lambda value: 0 if pd.isnull(value) else value)
        X['BsmtExposure'] = X['BsmtExposure'].apply(lambda value: 4 if value == 'Gd' else value)
        X['BsmtExposure'] = X['BsmtExposure'].apply(lambda value: 3 if value == 'Av' else value)
        X['BsmtExposure'] = X['BsmtExposure'].apply(lambda value: 2 if value == 'Mn' else value)
        X['BsmtExposure'] = X['BsmtExposure'].apply(lambda value: 1 if value == 'No' else value)

        # "BBsmtFinType1" - GLQ=6,ALQ=5,BLQ=4, Rec=3, LwQ=2,Unf=1
        X['BsmtFinType1'] = X['BsmtFinType1'].apply(lambda value: 0 if pd.isnull(value) else value)
        X['BsmtFinType1'] = X['BsmtFinType1'].apply(lambda value: 6 if value == 'GLQ' else value)
        X['BsmtFinType1'] = X['BsmtFinType1'].apply(lambda value: 5 if value == 'ALQ' else value)
        X['BsmtFinType1'] = X['BsmtFinType1'].apply(lambda value: 4 if value == 'BLQ' else value)
        X['BsmtFinType1'] = X['BsmtFinType1'].apply(lambda value: 3 if value == 'Rec' else value)
        X['BsmtFinType1'] = X['BsmtFinType1'].apply(lambda value: 2 if value == 'LwQ' else value)
        X['BsmtFinType1'] = X['BsmtFinType1'].apply(lambda value: 1 if value == 'Unf' else value)

        # BsmtFinSF1
        X['BsmtFinSF1'] = X['BsmtFinSF1'].apply(lambda value: np.nan if pd.isnull(value) else value)

        # "BBsmtFinType2" - GLQ=6,ALQ=5,BLQ=4, Rec=3, LwQ=2,Unf=1
        X['BsmtFinType2'] = X['BsmtFinType2'].apply(lambda value: 0 if pd.isnull(value) else value)
        X['BsmtFinType2'] = X['BsmtFinType2'].apply(lambda value: 6 if value == 'GLQ' else value)
        X['BsmtFinType2'] = X['BsmtFinType2'].apply(lambda value: 5 if value == 'ALQ' else value)
        X['BsmtFinType2'] = X['BsmtFinType2'].apply(lambda value: 4 if value == 'BLQ' else value)
        X['BsmtFinType2'] = X['BsmtFinType2'].apply(lambda value: 3 if value == 'Rec' else value)
        X['BsmtFinType2'] = X['BsmtFinType2'].apply(lambda value: 2 if value == 'LwQ' else value)
        X['BsmtFinType2'] = X['BsmtFinType2'].apply(lambda value: 1 if value == 'Unf' else value)

        # BsmtFinSF2
        X['BsmtFinSF2'] = X['BsmtFinSF2'].apply(lambda value: np.nan if pd.isnull(value) else value)

        # BsmtUnfSF
        X['BsmtUnfSF'] = X['BsmtUnfSF'].apply(lambda value: 0 if pd.isnull(value) else value)

        # TotalBsmtSF
        X['TotalBsmtSF'] = X['TotalBsmtSF'].apply(lambda value: 0 if pd.isnull(value) else value)

        # "Heating" - do nothing

        # "HeatingQC" - Ex=4, Gd=3, TA=2, Fa=1, Po=0,
        X['HeatingQC'] = X['HeatingQC'].apply(lambda value: self.__convert_str_to_int(value))

        # TotalBsmtSF
        X['CentralAir'] = X['CentralAir'].apply(lambda value: 1 if value == 'Y' else 0)

        # "Electrical" - SBrkr=4, FuseA=3,FuseF=2,FuseP=1,Mix=0
        X['Electrical'] = X['Electrical'].apply(lambda value: np.nan if pd.isnull(value) else value)
        X['Electrical'] = X['Electrical'].apply(lambda value: 4 if value == 'SBrkr' else value)
        X['Electrical'] = X['Electrical'].apply(lambda value: 3 if value == 'FuseA' else value)
        X['Electrical'] = X['Electrical'].apply(lambda value: 2 if value == 'FuseF' else value)
        X['Electrical'] = X['Electrical'].apply(lambda value: 1 if value == 'FuseP' else value)
        X['Electrical'] = X['Electrical'].apply(lambda value: 0 if value == 'Mix' else value)

        # 1stFlrSF
        X['1stFlrSF'] = X['1stFlrSF'].apply(lambda value: np.nan if pd.isnull(value) else value)

        # 2ndFlrSF
        X['2ndFlrSF'] = X['2ndFlrSF'].apply(lambda value: np.nan if pd.isnull(value) else value)

        # LowQualFinSF
        X['LowQualFinSF'] = X['LowQualFinSF'].apply(lambda value: np.nan if pd.isnull(value) else value)

        # GrLivArea
        X['GrLivArea'] = X['GrLivArea'].apply(lambda value: np.nan if pd.isnull(value) else value)

        # BsmtFullBath
        X['BsmtFullBath'] = X['BsmtFullBath'].apply(lambda value: np.nan if pd.isnull(value) else value)

        # BsmtHalfBath
        X['BsmtHalfBath'] = X['BsmtHalfBath'].apply(lambda value: np.nan if pd.isnull(value) else value)

        # FullBath
        X['FullBath'] = X['FullBath'].apply(lambda value: np.nan if pd.isnull(value) else value)

        # HalfBath
        X['HalfBath'] = X['HalfBath'].apply(lambda value: np.nan if pd.isnull(value) else value)

        # BedroomAbvGr
        X['BedroomAbvGr'] = X['BedroomAbvGr'].apply(lambda value: np.nan if pd.isnull(value) else value)

        # KitchenAbvGr
        X['KitchenAbvGr'] = X['KitchenAbvGr'].apply(lambda value: np.nan if pd.isnull(value) else value)

        # "KitchenQual" - Ex=4, Gd=3, TA=2, Fa=1, Po=0,
        X['KitchenQual'] = X['KitchenQual'].apply(lambda value: self.__convert_str_to_int(value))

        # TotRmsAbvGrd
        X['TotRmsAbvGrd'] = X['TotRmsAbvGrd'].apply(lambda value: np.nan if pd.isnull(value) else value)

        # "Functional" - Typ=7, Min1=6, Min2=5, Mod=4,Maj1=3, Maj2=2, Sev=1, Sal=0
        X['Functional'] = X['Functional'].apply(lambda value: np.nan if pd.isnull(value) else value)
        X['Functional'] = X['Functional'].apply(lambda value: 7 if value == 'Typ' else value)
        X['Functional'] = X['Functional'].apply(lambda value: 6 if value == 'Min1' else value)
        X['Functional'] = X['Functional'].apply(lambda value: 5 if value == 'Min2' else value)
        X['Functional'] = X['Functional'].apply(lambda value: 4 if value == 'Mod' else value)
        X['Functional'] = X['Functional'].apply(lambda value: 3 if value == 'Maj1' else value)
        X['Functional'] = X['Functional'].apply(lambda value: 2 if value == 'Maj2' else value)
        X['Functional'] = X['Functional'].apply(lambda value: 1 if value == 'Sev' else value)
        X['Functional'] = X['Functional'].apply(lambda value: 0 if value == 'Sal' else value)

        # Fireplaces
        X['Fireplaces'] = X['Fireplaces'].apply(lambda value: 0 if pd.isnull(value) else value)

        # "FireplaceQu" - Ex=4, Gd=3, TA=2, Fa=1, Po=0, NA=-1
        X['FireplaceQu'] = X['FireplaceQu'].apply(lambda value: self.__convert_str_to_int(value))
        X['FireplaceQu'] = X['FireplaceQu'].apply(lambda value: -1 if pd.isnull(value) else value)

        # GarageType
        X['GarageType'] = X['GarageType'].apply(lambda value: "NONE" if pd.isnull(value) else value)

        # "GarageYrBlt"
        X['GarageAge'] = X['YrSold'] - X['GarageYrBlt']

        # "GarageFinish" - Fin=3, RFn=2,Unf=1
        X['GarageFinish'] = X['GarageFinish'].apply(lambda value: 0 if pd.isnull(value) else value)
        X['GarageFinish'] = X['GarageFinish'].apply(lambda value: 3 if value == 'Fin' else value)
        X['GarageFinish'] = X['GarageFinish'].apply(lambda value: 2 if value == 'RFn' else value)
        X['GarageFinish'] = X['GarageFinish'].apply(lambda value: 1 if value == 'Unf' else value)

        # GarageCars
        X['GarageCars'] = X['GarageCars'].apply(lambda value: 0 if pd.isnull(value) else value)

        # GarageArea
        X['GarageArea'] = X['GarageArea'].apply(lambda value: 0 if pd.isnull(value) else value)

        # "GarageQual" - Ex=4, Gd=3, TA=2, Fa=1, Po=0,NA=-1
        X['GarageQual'] = X['GarageQual'].apply(lambda value: self.__convert_str_to_int(value))
        X['GarageQual'] = X['GarageQual'].apply(lambda value: -1 if pd.isnull(value) else value)

        # "GarageCond" - Ex=4, Gd=3, TA=2, Fa=1, Po=0,NA=-1
        X['GarageCond'] = X['GarageCond'].apply(lambda value: self.__convert_str_to_int(value))
        X['GarageCond'] = X['GarageCond'].apply(lambda value: -1 if pd.isnull(value) else value)

        # "PavedDrive" - Y=2,P=1,N=0
        X['PavedDrive'] = X['PavedDrive'].apply(lambda value: np.nan if pd.isnull(value) else value)
        X['PavedDrive'] = X['PavedDrive'].apply(lambda value: 2 if value == 'Y' else value)
        X['PavedDrive'] = X['PavedDrive'].apply(lambda value: 1 if value == 'P' else value)
        X['PavedDrive'] = X['PavedDrive'].apply(lambda value: 0 if value == 'N' else value)

        # WoodDeckSF
        X['WoodDeckSF'] = X['WoodDeckSF'].apply(lambda value: 0 if pd.isnull(value) else value)

        # OpenPorchSF
        X['OpenPorchSF'] = X['OpenPorchSF'].apply(lambda value: 0 if pd.isnull(value) else value)

        # EnclosedPorch
        X['EnclosedPorch'] = X['EnclosedPorch'].apply(lambda value: 0 if pd.isnull(value) else value)

        # 3SsnPorch
        X['3SsnPorch'] = X['3SsnPorch'].apply(lambda value: 0 if pd.isnull(value) else value)

        # ScreenPorch
        X['ScreenPorch'] = X['ScreenPorch'].apply(lambda value: 0 if pd.isnull(value) else value)

        # PoolArea
        X['PoolArea'] = X['PoolArea'].apply(lambda value: 0 if pd.isnull(value) else value)

        # "PoolQC" - Ex=4, Gd=3, TA=2, Fa=1, Po=0
        X['PoolQC'] = X['PoolQC'].apply(lambda value: self.__convert_str_to_int(value))
        X['PoolQC'] = X['PoolQC'].apply(lambda value: 0 if pd.isnull(value) else value)

        # "Fence" - GdPrv=4,
        X['Fence'] = X['Fence'].apply(lambda value: 0 if pd.isnull(value) else value)
        X['Fence'] = X['Fence'].apply(lambda value: 4 if value == 'GdPrv' else value)
        X['Fence'] = X['Fence'].apply(lambda value: 3 if value == 'MnPrv' else value)
        X['Fence'] = X['Fence'].apply(lambda value: 2 if value == 'GdWo' else value)
        X['Fence'] = X['Fence'].apply(lambda value: 1 if value == 'MnWo' else value)

        # MiscFeature
        X['MiscFeature'] = X['MiscFeature'].apply(lambda value: 'Miss' if pd.isnull(value) else value)

        # MiscVal
        X['MiscVal'] = X['MiscVal'].apply(lambda value: 0 if pd.isnull(value) else value)

        # MoSold
        X['MoSold'] = X['MoSold'].apply(lambda value: np.nan if pd.isnull(value) else 'month_' + str(value))

        # YrSold
        X['YrSold'] = X['YrSold'].apply(lambda value: np.nan if pd.isnull(value) else value)

        # SaleType
        X['SaleType'] = X['SaleType'].apply(lambda value: np.nan if pd.isnull(value) else value)

        # SaleCondition
        X['SaleCondition'] = X['SaleCondition'].apply(lambda value: np.nan if pd.isnull(value) else value)

        return X

    def __convert_str_to_int(self, str_val):
        if str_val == 'Ex':
            return 4
        elif str_val == 'Gd':
            return 3
        elif str_val == 'TA':
            return 2
        elif str_val == 'FA':
            return 1
        elif str_val == 'Po':
            return 0
        else:
            return np.nan