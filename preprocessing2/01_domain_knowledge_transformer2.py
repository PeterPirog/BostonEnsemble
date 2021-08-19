import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin


class DomainKnowledgeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, remove_original=True):
        super().__init__()
        self.remove_original = remove_original

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Remove index column
        X = X.drop(['Id'], axis=1)

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
        X['MiscFeature'] = X['MiscFeature'].apply(lambda value: np.nan if pd.isnull(value) else value)

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


if __name__ == '__main__':
    verbose = False

    # define path to the project directory
    base_path = Path(__file__).parent.parent

    # make all dataframe columns visible
    pd.set_option('display.max_columns', None)

    df_train = pd.read_csv('/home/peterpirog/PycharmProjects/BostonEnsemble/data_files/train.csv')
    df_test = pd.read_csv('/home/peterpirog/PycharmProjects/BostonEnsemble/data_files/test.csv')

    X = df_train.drop(['SalePrice'], axis='columns').copy()
    y = np.log1p(df_train['SalePrice']).copy()

    X_test = df_test.copy()

    dkt = DomainKnowledgeTransformer()

    dkt.fit_transform(X=X, y=y)
    y_out = dkt.transform(X=X_test)

    print(y_out.head(10))
    #print(y_out.info())
    # print(df_out_train.describe())
