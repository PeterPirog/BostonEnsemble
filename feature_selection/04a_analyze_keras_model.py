
import pandas as pd


if __name__ == "__main__":

    # Get all fetures in dataframe
    pd.set_option('display.max_columns', None)
    df = pd.read_csv('forward_keras_features.csv')
    print(df.head())
    df.to_excel('forward_keras_features.xlsx')
