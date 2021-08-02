
import tensorflow as tf
import sys
import numpy as np

from pathlib import Path
import pandas as pd
import matplotlib.pylab as plt

if __name__ == "__main__":
    base_path = Path(__file__).parent.parent
    np.set_printoptions(threshold=sys.maxsize)
    # Get all fetures in dataframe
    df = pd.read_csv((base_path / "./data_files/encoded_train_X_data.csv").resolve())
    y = df['SalePrice_log1']
    X = df.drop(['Id', 'SalePrice', 'SalePrice_log1'], axis=1).copy()
    features=X.columns
    n_features=len(features)
    X = X.to_numpy()
    y = y.to_numpy()

    model=tf.keras.models.load_model('model_1L.h5')
    w=model.get_layer(name='dense_layer').get_weights()[0]
    wo=model.get_layer(name='output_layer').get_weights()[0]

    w=np.array(w)
    wo = np.array(wo)
    w=np.abs(w)
    wo=np.abs(wo)

    print(f'w={w}, {w.shape}')
    print(f'wo={wo}, {wo.shape}')

    n_neurons=wo.shape[0]
    print(f'n_neurons={n_neurons}')
    c=w.copy()
    r = w.copy()

    #calculate c array
    for i in range(n_features):
        for j in range(n_neurons):
            c[i,j]=w[i,j]*wo[j,0]


    for i in range(n_features):
        for j in range(n_neurons):
            #sum = np.sum(c[:,j])
            r[i,j]=c[i,j]/np.sum(c[:,j])


    #print(f'c={c}, {c.shape}')
    #print(f'r={r}, {r.shape}')

    #print('sum=',np.sum(r,axis=0))
    S=np.sum(r,axis=1)
    S=100*S/np.sum(S)
    #print('S=', S)
    d={}
    for idx,feature in enumerate(features):
        d[feature]=S[idx]
    sorted_d=dict(sorted(d.items(), key=lambda item: item[1],reverse=True))
    #print(sorted_d)
    feature_list=[]
    values_list=[]
    values_additive=[]
    print('\n Feature analysis:')
    for i,key in enumerate(sorted_d):
        print(i+1,key,sorted_d[key])
        feature_list.append(key)
        values_list.append(sorted_d[key])
        values_additive.append(np.sum(values_list[:i+1]))

    print('feature_list',feature_list)
    print('values_additive',values_additive)




    #plt.plot( feature_list,values_list)
    plt.plot( feature_list,values_additive)

    plt.xticks(rotation=90,fontsize=5)
    plt.show()


#feature_list  for 3 neurons ['_KitchenQual', '_GarageType_Detchd', '_Neighborhood_2', '_OverallQual', '_HouseStyle_3', '_WoodDeckSF', '_GrLivArea', '_Exterior_VinylSd', '_BsmtUnfSF', '_HouseStyle_2', '_MSSubClass_2', '_Foundation_1', '_MSSubClass_3', '_HeatingQC', '_Foundation_3', '_HouseStyle_1', '_MSSubClass_1', '_Neighborhood_5', '_TotalBsmtSF', '_BsmtQual', '_GarageQual', '_CentralAir', '_OverallCond', '_Foundation_2', '_ScreenPorch', '_MSZoning_3', '_GarageArea', '_BuildingAge', '_SaleCondition_Abnorml', '_HouseStyle_4', '_BldgType_2', '_MSZoning_1', '_BsmtFinSF1', '_BldgType_1', '_SaleCondition_Normal', '_Electrical', '_PavedDrive', '_GarageType_Attchd', '_GarageFinish', '_Neighborhood_9', '_LotArea', '_LandContour_2', '_LotFrontage', '_YearRemodAdd', '_Exterior_WdSdng', '_MasVnrType_Stone', '_HalfBath', '_EnclosedPorch', '_Exterior_Plywood', '_FireplaceQu', '_Functional', '_FullBath', '_LotConfig_4', '_BsmtCond', '_SaleCondition_Partial', '_RoofStyle_1', '_BsmtExposure', '_MasVnrType_BrkFace', '_Neighborhood_8', '_Alley', '_Exterior_MetalSd', '_BsmtFinSF2', '_Exterior_HdBoard', '_Neighborhood_1', '_LotConfig_1', '_Fence', '_SaleType_WD', '_BldgType_3', '_BsmtFinType1', '_BsmtFullBath', '_LotShape', '_BedroomAbvGr', '_OpenPorchSF', '_LotConfig_3', '_MasVnrArea', '_ExterQual', '_GarageType_BuiltIn', '_YrSold', '_QuarterSold', '_ExterCond']
#feature_list for 20 neurons ['_BldgType_2', '_BldgType_3', '_BldgType_1', '_BuildingAge', '_GrLivArea', '_OverallQual', '_OverallCond', '_MSZoning_3', '_KitchenQual', '_Neighborhood_2', '_TotalBsmtSF', '_BsmtFinSF1', '_LotArea', '_SaleCondition_Abnorml', '_Neighborhood_9', '_GarageArea', '_OpenPorchSF', '_CentralAir', '_FireplaceQu', '_PavedDrive', '_YearRemodAdd', '_BsmtQual', '_Neighborhood_8', '_Neighborhood_5', '_HalfBath', '_BsmtFullBath', '_Functional', '_MSSubClass_3', '_BsmtExposure', '_HouseStyle_2', '_Electrical', '_GarageQual', '_MSZoning_1', '_Foundation_2', '_HeatingQC', '_FullBath', '_YrSold', '_ScreenPorch', '_WoodDeckSF', '_BsmtUnfSF', '_MasVnrArea', '_LotFrontage', '_SaleType_WD', '_GarageType_Attchd', '_GarageFinish', '_LotConfig_1', '_HouseStyle_3', '_RoofStyle_1', '_Neighborhood_1', '_Exterior_HdBoard', '_HouseStyle_1', '_Exterior_WdSdng', '_ExterQual', '_MSSubClass_1', '_Foundation_3', '_EnclosedPorch', '_GarageType_Detchd', '_BedroomAbvGr', '_LotConfig_4', '_MasVnrType_Stone', '_Exterior_MetalSd', '_BsmtFinType1', '_GarageType_BuiltIn', '_MSSubClass_2', '_SaleCondition_Normal', '_BsmtCond', '_Exterior_VinylSd', '_SaleCondition_Partial', '_BsmtFinSF2', '_MasVnrType_BrkFace', '_HouseStyle_4', '_QuarterSold', '_Alley', '_Exterior_Plywood', '_LotShape', '_LandContour_2', '_Fence', '_Foundation_1', '_ExterCond', '_LotConfig_3']
