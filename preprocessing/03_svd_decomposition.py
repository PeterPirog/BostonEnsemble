import modin.pandas as pd
import numpy as np
from pathlib import Path


if __name__ == "__main__":
    # define path to the project directory
    base_path = Path(__file__).parent.parent

    df_train = pd.read_csv((base_path / "./data_files/encoded_train_X_data.csv").resolve())
    df_train= df_train.drop(['SalePrice','Id','SalePrice_log1'], axis=1)
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