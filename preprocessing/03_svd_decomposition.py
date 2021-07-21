import pandas as pd
import numpy as np
from pathlib import Path


if __name__ == "__main__":
    # define path to the project directory
    base_path = Path(__file__).parent.parent

    df_train = pd.read_csv((base_path / "./data_files/encoded_train_X_data.csv").resolve())
    X=df_train.to_numpy()
    #print(X)
    u, s, vh = np.linalg.svd(X, full_matrices=True)

    #print(s)
    ssq = np.sum(s ** 2)

    power=s**2/ssq
    print(100*power)