from sktools import QuantileEncoder
import pandas as pd
pd.set_option('display.max_columns', None)
from sklearn.datasets import load_boston
bunch = load_boston()
y = bunch.target
X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
print(X.head())
enc = QuantileEncoder(cols=['CHAS', 'RAD'],quantile=0.5,m=1).fit(X, y)
numeric_dataset = enc.transform(X)
#print(numeric_dataset.info())
print(numeric_dataset.head())