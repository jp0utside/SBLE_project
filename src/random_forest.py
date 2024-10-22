import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X, y, n_estimators=100, random_state=3):
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    clf.fit(X, y)

    print("Prediction Probabilities: ")
    print(clf.predict_proba(X))

    print()

    print("Feature Importance Scores: ")
    print(clf.feature_importances_)
    
    return clf