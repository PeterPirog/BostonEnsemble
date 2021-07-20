
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from preprocessing.Transformers import IterativeImputerDf

if __name__ == "__main__":

    X_train=pd.DataFrame([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]],columns=['aaa','bbb','ccc'])
    X_train = pd.DataFrame([[7, 2, 3], [4, np.nan, 6], [10, 5, 9],[1,2,3]], columns=['aaa', 'bbb', 'ccc'])
    X_test=pd.DataFrame([[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9], [10, 4.5, np.nan]]], columns=['aaa','bbb','ccc'])
    print(X_train.head())

    imp_mean = IterativeImputerDf(random_state=0,dataframe_as_output=True)
    out_train=imp_mean.fit_transform(X_train)
    out_test = imp_mean.transform(X_test)


    print(out_test)