#https://machinelearningmastery.com/multivariate-adaptive-regression-splines-mars-in-python/
#https://github.com/scikit-learn-contrib/py-earth/blob/master/pyearth/earth.py
import pandas as pd
from joblib import dump, load


from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from ensemble_tools import FeatureByNameSelector, Validator,read_config_files
#pip install git+https://github.com/scikit-learn-contrib/py-earth@v0.2dev


# check pyearth version
from pyearth import Earth

#pip install git+https://github.com/scikit-learn-contrib/py-earth@v0.2dev

if __name__ == "__main__":
    conf_global = read_config_files(configuration_name='conf_global')
    conf_ridge = read_config_files(configuration_name='conf_ridge')

    df = pd.read_csv(conf_global['encoded_train_data'])

    all_features= ['_BldgType_2', '_BldgType_1', '_BldgType_3', '_GrLivArea',
       '_MSSubClass_3', '_OverallQual', '_BuildingAge', '_TotalBsmtSF',
       '_Functional', '_CentralAir', '_Electrical', '_SaleCondition_Abnorml',
       '_RoofStyle_1', '_LotArea', '_GarageArea', '_KitchenQual',
       '_OverallCond', '_Neighborhood_9', '_SaleType_WD', '_ScreenPorch',
       '_BsmtExposure', '_ExterQual', '_BsmtUnfSF', '_Foundation_2',
       '_HouseStyle_2', '_HouseStyle_3', '_LotConfig_4', '_GarageType_BuiltIn',
       '_FullBath', '_Neighborhood_1', '_FireplaceQu', '_BsmtQual',
       '_SaleCondition_Normal', '_BsmtFinType1', '_PavedDrive',
       '_Foundation_3', '_MSZoning_1', '_Neighborhood_5', '_HeatingQC',
       '_YrSold', '_HalfBath', '_YearRemodAdd', '_GarageFinish',
       '_HouseStyle_1', '_BsmtFinSF2', '_WoodDeckSF', '_Exterior_VinylSd',
       '_MSSubClass_1', '_GarageType_Attchd', '_LotFrontage',
       '_Exterior_HdBoard', '_HouseStyle_4', '_MasVnrType_BrkFace',
       '_Exterior_Plywood', '_GarageQual', '_MasVnrType_Stone',
       '_LandContour_2', '_BsmtFullBath', '_LotShape', '_Exterior_WdSdng',
       '_Neighborhood_8', '_Fence', '_LotConfig_1', '_Alley',
       '_Exterior_MetalSd', '_EnclosedPorch', '_LotConfig_3', '_BsmtCond',
       '_MasVnrArea', '_SaleCondition_Partial', '_GarageType_Detchd',
       '_MSZoning_3', '_ExterCond', '_Neighborhood_2', '_QuarterSold',
       '_BsmtFinSF1', '_BedroomAbvGr', '_OpenPorchSF', '_Foundation_1',
       '_MSSubClass_2']

    X = df[all_features]

    y = df[conf_global['target_label']]

    #Select only 'n_features' labels for X dataframe
    fsel = FeatureByNameSelector(all_features=conf_global['all_features'],
                                 n_features=conf_ridge['n_features'])

    model = Earth()

    pipe = Pipeline([
        ('fsel', fsel),
        ('model', model),
    ])
    #train model with parameters defined in conf_ridge,json file
    model.fit(X=X, y=y)

    #Save pipeline or model in joblib file
    dump(model, filename=conf_ridge['output_file'])
    v = Validator(model_or_pipeline=pipe, X=X, y=y, n_splits=10, n_repeats=5, random_state=1,
                  scoring='neg_root_mean_squared_error',model_config_dict=conf_ridge)
    v.run()

    #print(pipe.predict(X[:5]))
    print(model.summary())

