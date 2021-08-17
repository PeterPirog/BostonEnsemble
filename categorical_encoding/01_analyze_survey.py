# https://github.com/david26694/QE_experiments/blob/master/DecisionTreePlot_ohe_te_qe.ipynb
import pandas as pd
#import modin.pandas as pd

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt

from category_encoders.m_estimate import MEstimateEncoder
from category_encoders.one_hot import OneHotEncoder
import sktools
from xgboost import XGBRegressor
#import ray
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer

if __name__ == "__main__":
    #ray.init()
    df = pd.read_csv(
        "/home/peterpirog/PycharmProjects/BostonEnsemble/categorical_encoding/developer_survey_2020/survey_results_public.csv")
    features = ['MainBranch', 'Hobbyist', 'Age', 'Age1stCode', 'CompFreq', 'Country', 'CurrencyDesc',
                'CurrencySymbol', 'DatabaseDesireNextYear', 'DatabaseWorkedWith',
                'DevType', 'EdLevel', 'Employment', 'Ethnicity', 'Gender', 'JobFactors',
                'JobSat', 'JobSeek', 'LanguageDesireNextYear', 'LanguageWorkedWith',
                'MiscTechDesireNextYear', 'MiscTechWorkedWith',
                'NEWCollabToolsDesireNextYear', 'NEWCollabToolsWorkedWith', 'NEWDevOps',
                'NEWDevOpsImpt', 'NEWEdImpt', 'NEWJobHunt', 'NEWJobHuntResearch',
                'NEWLearn', 'NEWOffTopic', 'NEWOnboardGood', 'NEWOtherComms',
                'NEWOvertime', 'NEWPurchaseResearch', 'NEWPurpleLink', 'NEWSOSites',
                'NEWStuck', 'OpSys', 'OrgSize', 'PlatformDesireNextYear',
                'PlatformWorkedWith', 'PurchaseWhat', 'Sexuality', 'SOAccount',
                'SOComm', 'SOPartFreq', 'SOVisitFreq', 'SurveyEase', 'SurveyLength',
                'Trans', 'UndergradMajor', 'WebframeDesireNextYear',
                'WebframeWorkedWith', 'WelcomeChange', 'WorkWeekHrs', 'YearsCode',
                'YearsCodePro']

    features=['Country']
    target = 'ConvertedComp'  # 'CompTotal', 'ConvertedComp'
    X = df[features]
    y = df[target]


    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)

    print(y_tr)

    te = MEstimateEncoder()
    pe = sktools.encoders.QuantileEncoder(quantile=0.5, m=1,handle_missing='return_nan', handle_unknown='return_nan')
    ohe = OneHotEncoder(handle_missing='return_nan', handle_unknown='return_nan')
    imp=IterativeImputer(min_value=-np.inf,  # values from 0 to 1 for categorical for numeric
                           max_value=np.inf,
                           random_state=42,
                           initial_strategy='median',
                           max_iter=10,
                           tol=0.01,
                           verbose=2)

    #X_tr_ohe = ohe.fit_transform(X_tr)
    #X_tr_te = te.fit_transform(X_tr, y_tr)

    X_tr_pe = pe.fit_transform(X_tr, y_tr)
    print(X_tr_pe)
    imp.fit(X=X_tr_pe,y=None)
    X_tr_pe = imp.transform(X_tr_pe)



    """
    model = XGBRegressor(n_estimators=50,
                         max_depth=6,
                         eta=0.1,
                         subsample=1,
                         colsample_bytree=1, n_jobs=-1)
    
    # ONE HOT

    model.fit(X_tr_ohe, y_tr)

    print(mean_absolute_error(model.predict(X_tr_ohe), y_tr))
    print(mean_absolute_error(model.predict(ohe.transform(X_te)), y_te))

    plt.figure()
    plot_tree(model, filled=True)
    plt.savefig('tree_ohe.eps', format='eps')
    plt.show()
  

    # Target Encodings
    model.fit(X_tr_te, y_tr)

    print(mean_absolute_error(model.predict(X_tr_te), y_tr))
    print(mean_absolute_error(model.predict(te.transform(X_te)), y_te))

    plt.figure()
    plot_tree(model, filled=True)
    plt.savefig('tree_te.eps', format='eps')
    plt.show()


    # QUANTILE
    model.fit(X_tr_pe, y_tr)

    print(mean_absolute_error(model.predict(X_tr_pe), y_tr))
    print(mean_absolute_error(model.predict(pe.transform(X_te)), y_te))

    plt.figure()
    plot_tree(model, filled=True)
    plt.show()

    """