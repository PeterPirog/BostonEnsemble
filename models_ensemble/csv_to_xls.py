import pandas as pd

df = pd.read_csv('/home/peterpirog/PycharmProjects/BostonEnsemble/feature_selection/forward_keras_features.csv')
df.to_excel('/home/peterpirog/PycharmProjects/BostonEnsemble/feature_selection/forward_keras_features_v2.xlsx')