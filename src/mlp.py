import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.base import clone

class MLP(BaseEstimator, ClassifierMixin):
    _mlp_params = {
        'hidden_layer_sizes', 'activation', 'solver', 'alpha', 'batch_size',
        'learning_rate', 'learning_rate_init', 'power_t', 'max_iter', 'shuffle',
        'random_state', 'tol', 'verbose', 'warm_start', 'momentum',
        'nesterovs_momentum', 'early_stopping', 'validation_fraction',
        'beta_1', 'beta_2', 'epsilon', 'n_iter_no_change', 'max_fun'
    }

    def __init__(self, scaler = None, pca = None, features = [], pca_features = [], **mlp_params):
        self.scaler = scaler
        self.pca = pca
        self.features = features
        self.pca_features = pca_features

        for param, value in mlp_params.items():
            if param in self._mlp_params:
                setattr(self, param, value)

        self.mlp = MLPClassifier(**mlp_params)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator.
        
        Override BaseEstimator.get_params to declare valid parameters.
        """
        # Get parameters from parent class
        params = super().get_params(deep=deep)
        
        # Add MLPClassifier parameters
        if deep:
            mlp_params = self.mlp.get_params(deep=deep)
            for param in self._mlp_params:
                if param in mlp_params:
                    params[param] = mlp_params[param]
        
        return params
    
    def set_params(self, **params):
        """Set the parameters of this estimator."""
        valid_params = self.get_params(deep=True)
        mlp_params = {}
        
        for param, value in params.items():
            if param in valid_params:
                if param in self._mlp_params:
                    mlp_params[param] = value
                else:
                    setattr(self, param, value)
        
        if mlp_params:
            self.mlp.set_params(**mlp_params)
        
        return self


    def prep(self, X_train, X_test):
        all_feats = list(set(self.features + self.pca_features))
        X_train_new = X_train.copy()[all_feats]
        X_test_new = X_test.copy()[all_feats]

        pca_idxs = [list(X_train_new.columns).index(feat) for feat in self.pca_features]
        feat_idxs = [list(X_train_new.columns).index(feat) for feat in self.features]

        X_train_new = X_train_new.to_numpy()
        X_test_new = X_test_new.to_numpy()

        if self.scaler is not None:
            temp_scaler = clone(self.scaler)
            X_train_new = temp_scaler.fit_transform(X_train_new)
            X_test_new = temp_scaler.transform(X_test_new)

        if self.pca is not None:
            temp_pca = clone(self.pca)
            X_train_pca = X_train_new[:, pca_idxs].copy()
            X_test_pca = X_test_new[:, pca_idxs].copy()

            X_train_new = X_train_new[:, feat_idxs]
            X_test_new = X_test_new[:, feat_idxs]

            X_train_pca = temp_pca.fit_transform(X_train_pca)
            X_test_pca = temp_pca.transform(X_test_pca)

            X_train_new = np.append(X_train_new, X_train_pca, 1)
            X_test_new = np.append(X_test_new, X_test_pca, 1)

        return X_train_new, X_test_new

    def prep_train_data(self, X_train):
        all_feats = list(set(self.features + self.pca_features))
        X_train_new = X_train.copy()[all_feats]

        pca_idxs = [list(X_train_new.columns).index(feat) for feat in self.pca_features]
        feat_idxs = [list(X_train_new.columns).index(feat) for feat in self.features]

        X_train_new = X_train_new.to_numpy()

        if self.scaler is not None:
            X_train_new = self.scaler.fit_transform(X_train_new)
        
        if self.pca is not None:
            X_train_pca = X_train_new[:, pca_idxs].copy()
            X_train_new = X_train_new[:, feat_idxs]

            X_train_pca = self.pca.fit_transform(X_train_pca)

            X_train_new = np.append(X_train_new, X_train_pca, 1)
        
        return X_train_new

    def prep_test_data(self, X_test):
        all_feats = list(set(self.features + self.pca_features))
        X_test_new = X_test.copy()[all_feats]

        pca_idxs = [list(X_test_new.columns).index(feat) for feat in self.pca_features]
        feat_idxs = [list(X_test_new.columns).index(feat) for feat in self.features]

        X_test_new = X_test_new.to_numpy()

        if self.scaler is not None:
            X_test_new = self.scaler.transform(X_test_new)
        
        if self.pca is not None:
            X_test_pca = X_test_new[:, pca_idxs].copy()
            X_test_new = X_test_new[:, feat_idxs]

            X_test_pca = self.pca.transform(X_test_pca)

            X_test_new = np.append(X_test_new, X_test_pca, 1)
        
        return X_test_new

    """
    X: pd.DataFrame of shape = (n_samples, n_columns)
    y: pd.DataFrame of shape = (n_samples,)
    """
    def fit(self, X, y):
        X_train = self.prep_train_data(X)
        self.mlp.fit(X_train, y)

        self.classes_ = self.mlp.classes_
        self.n_layers_ = self.mlp.n_layers_
        self.n_outputs_ = self.mlp.n_outputs_
        self.n_features_in_ = self.mlp.n_features_in_
        self.feature_names_in_ = getattr(self.mlp, 'feature_names_in_', None)
        
        return self
    
    """
    X: pd.DataFrame of shape = (n_samples, n_columns)
    y: pd.DataFrame of shape = (n_samples,)
    """
    def predict(self, X):
        X_test = self.prep_test_data(X)
        return self.mlp.predict(X_test)
    
    def predict_proba(self, X):
        X_test = self.prep_test_data(X)
        return self.mlp.predict_proba(X)


    
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

            print("X_train shape: ", X_train.shape)
            print("X_test shape: ", X_test.shape)

            clf = clone(self.mlp)

            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            scores.append(acc)
        
        return scores