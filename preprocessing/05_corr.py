import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



if __name__ == "__main__":



    # define path to the project directory
    base_path = Path(__file__).parent.parent

    df_train = pd.read_csv((base_path / "./data_files/encoded_train_X_data.csv").resolve())
    df_train= df_train.drop(['SalePrice','Id'], axis=1)
    plt.matshow(df_train.corr())
    #plt.show()

    c=df_train[df_train.columns[1:]].corr()['SalePrice_log1'][:]
    c.to_excel("output_corr.xlsx")

    df=c.to_frame().reset_index()
    print('Type=',type(df))
    print('columns=', df.columns)

    df['Feature']=df['index']
    df['Corr'] = df['SalePrice_log1']
    df['Abs_Corr']=abs(df['Corr'])
    df=df.drop(['index','SalePrice_log1'],axis=1)
    df=df.sort_values(by=['Abs_Corr'],ascending=False)

    arr = df["Feature"].to_list()
    del arr[0]

    #df.rename({1: 2, 2: 4}, axis='index')
    print('List len=',len(arr))
    print(arr)
    """
    columns=df_train.columns
    print('Columns:',columns)
    X=df_train.to_numpy()
    #print(X)
    u, s, vh = np.linalg.svd(X, full_matrices=True)
    df_u=pd.DataFrame(u)
    df_s = pd.DataFrame(s)
    df_vh = pd.DataFrame(vh,columns=columns)

    df_u.to_excel(
        'svd_analysis_u.xlsx',
        sheet_name='output_u',
        index=False)
    df_s.to_excel(
        'svd_analysis_s.xlsx',
        sheet_name='output_s',
        index=False)
    df_vh.to_excel(
        'svd_analysis_vh.xlsx',
        sheet_name='output_vh',
        index=False)

    #print(s)
    ssq = np.sum(s ** 2)

    power=s**2/ssq
    print(100*power)
    """