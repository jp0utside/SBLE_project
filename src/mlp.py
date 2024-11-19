import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.base import clone

class MLP:
    def __init__(self, scaler = None, pca = None):
        self.scaler = scaler
        self.pca = pca
        self.clf = MLPClassifier()

    def prep(self, X_train, X_test):
        X_train_new = X_train.copy()
        X_test_new = X_test.copy()


        if self.scaler is not None:
            temp_scaler = clone(self.scaler)
            X_train_new = temp_scaler.fit_transform(X_train_new)
            X_test_new = temp_scaler.transform(X_test_new)

        if self.pca is not None:
            temp_pca = clone(self.pca)
            X_train_new = temp_pca.fit_transform(X_train_new)
            X_test_new = temp_pca.transform(X_test_new)

        return X_train_new, X_test_new
    
    def kfold(self, X, y, n_splits = 5):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state = 3)

        scores = []
        
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            # print("i: {}, (train_index, test_index) = ({}, {})".format(i, train_index, test_index))
            X_train = [X[i] for i in train_index].copy()
            y_train = [y[i] for i in train_index].copy()
            X_test = [X[i] for i in test_index].copy()
            y_test = [y[i] for i in test_index].copy()

            X_train = pd.concat(X_train)
            y_train = pd.concat(y_train)
            X_test = pd.concat(X_test)
            y_test = pd.concat(y_test)

            if self.scaler:
                new_scaler = clone(self.scaler)
                X_train = new_scaler.fit_transform(X_train)
                X_test = new_scaler.transform(X_test)
            if self.pca:
                new_pca = clone(self.pca)
                X_train = new_pca.fit_transform(X_train)
                X_test = new_pca.transform(X_test)

            clf = clone(self.clf)

            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            scores.append(acc)
        
        return scores

    def GroupKFold(self, X, y, groups, n_splits = 5):
        kf = GroupKFold(n_splits=n_splits)

        scores = []
        
        for i, (train_index, test_index) in enumerate(kf.split(X, y, groups)):
            # print("i: {}, (train_index, test_index) = ({}, {})".format(i, train_index, test_index))
            X_train = X.iloc[train_index].copy()
            y_train = y.iloc[train_index].copy()
            X_test = X.iloc[test_index].copy()
            y_test = y.iloc[test_index].copy()

            X_train, X_test = self.prep(X_train, X_test)

            clf = clone(self.clf)

            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            scores.append(acc)
        
        return scores