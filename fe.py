import re
import pickle

import pandas as pd
import numpy as np

from openfe import OpenFE, transform
from openfe.FeatureGenerator import Node

from sklearn.model_selection import StratifiedGroupKFold

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    
    X_train = pd.read_csv('Xtrain.csv').rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    Y_train = pd.read_csv('Ytrain.csv')

    cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=0xFACED)

    for (_, index) in cv.split(
            X_train.drop(["GameRulesetName"], axis=1),
            Y_train["utility_agent1"].astype(str),
            X_train["GameRulesetName"]
        ):
        X_train = X_train.iloc[index].reset_index()
        Y_train = Y_train.iloc[index].reset_index()
        break

    print(X_train.shape, Y_train.shape)

    ofe = OpenFE()
    
    features = ofe.fit(
        data=X_train.drop(["GameRulesetName"], axis=1),
        label=Y_train["utility_agent1"].astype(np.float64),
        group=X_train["GameRulesetName"],
        n_jobs=1
    )

    with open('feature.pickle', 'wb') as file:
        pickle.dump(features, file)

    train_x, _ = transform(
        X_train,
        X_train,
        features,
        n_jobs=1
    )

    train_x.to_csv('features.csv')