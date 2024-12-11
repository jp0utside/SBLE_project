import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.base import clone, BaseEstimator, ClassifierMixin

"""
Wrapper for sklearn RandomForestClassifier
Accommodates choosing different featuresets and pca features
Follows sklearn classifier standards in order to work with Gridsearch
Takes in data of shape (n_samples, n_features)
"""
class RandomForest(BaseEstimator, ClassifierMixin):
    _rf_params = {
        'n_estimators', 'criterion', 'max_depth', 'min_samples_split',
        'min_samples_leaf', 'min_weight_fraction_leaf', 'max_features',
        'max_leaf_nodes', 'min_impurity_decrease', 'bootstrap',
        'oob_score', 'n_jobs', 'random_state', 'verbose', 'warm_start',
        'class_weight', 'ccp_alpha', 'max_samples'
    }
    def __init__(self, scaler = None, pca = None, features = [], pca_features = [], **rf_params):
        self.scaler = scaler
        self.pca = pca
        self.features = features
        self.pca_features = pca_features
        
        for param, value in rf_params.items():
            if param in self._rf_params:
                setattr(self, param, value)
            
        self.rf = RandomForestClassifier(**rf_params)

    """
    Getter and Setter functions to override parent functions for RandomForestClassifier
    """
    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        
        if deep:
            rf_params = self.rf.get_params(deep=deep)
            for param in self._rf_params:
                if param in rf_params:
                    params[param] = rf_params[param]
        
        return params
    
    def set_params(self, **params):
        valid_params = self.get_params(deep=True)
        rf_params = {}
        
        for param, value in params.items():
            if param in valid_params:
                if param in self._rf_params:
                    rf_params[param] = value
                else:
                    setattr(self, param, value)
        
        if rf_params:
            self.rf.set_params(**rf_params)
        
        return self
    
    def prep_training_data(self, X_train):
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
    
    def prep_testing_data(self, X_test):
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
        X_train = pd.concat(X)
        y_train = pd.concat([pd.Series([y[i]]*frame.shape[0]) for i, frame in enumerate(X)])

        X_train = self.prep_training_data(X_train)
        y_train = np.array(y_train)

        self.rf.fit(X_train, y_train)

        self.classes_ = np.unique(y_train)
        self.n_features_in_ = X_train.shape[1]
        
        return self
    
    """
    X: pd.DataFrame of shape = (n_samples, n_columns)
    y: pd.DataFrame of shape = (n_samples,)
    """
    def predict(self, X):
        X_test = pd.concat(X)
        X_test = self.prep_testing_data(X_test)

        return self.rf.predict(X_test)
    
    def predict_proba(self, X):
        X_test = pd.concat(X)
        X_test = self.prep_testing_data(X_test)

        return self.rf.predict_proba(X_test)
    
"""
Scoring function to be used in Gridsearch
Allows for compatability of stratified kfold cross validation, as y values need to represent entire frame instead of being repeated for each row in the frame
model: trained RandomForest model used to predict labels
X: array of test data of shape (n_samples, n_features)
y: array of test targets
"""
def rf_prediction_scorer(model, X, y):
    y_pred = model.predict(X)
    y_true = np.array(pd.concat([pd.Series([y[i]]*frame.shape[0]) for i, frame in enumerate(X)]))
    # for i, frame in enumerate(X):
    #     y_true.extend([y[i] for it in range(frame.shape[0])])
    acc = accuracy_score(y_true, y_pred)
    return acc

    # def kfold(self, X, y, n_splits = 5):
    #     kf = KFold(n_splits=n_splits, shuffle=True, random_state = 3)

    #     scores = []
        
    #     for i, (train_index, test_index) in enumerate(kf.split(X)):
    #         # print("i: {}, (train_index, test_index) = ({}, {})".format(i, train_index, test_index))
    #         X_train = [X[i] for i in train_index].copy()
    #         y_train = [y[i] for i in train_index].copy()
    #         X_test = [X[i] for i in test_index].copy()
    #         y_test = [y[i] for i in test_index].copy()

    #         X_train = pd.concat(X_train)
    #         y_train = pd.concat(y_train)
    #         X_test = pd.concat(X_test)
    #         y_test = pd.concat(y_test)

    #         if self.scaler:
    #             new_scaler = clone(self.scaler)
    #             X_train = new_scaler.fit_transform(X_train)
    #             X_test = new_scaler.transform(X_test)
    #         if self.pca:
    #             new_pca = clone(self.pca)
    #             X_train = new_pca.fit_transform(X_train)
    #             X_test = new_pca.transform(X_test)

    #         clf = clone(self.clf)

    #         clf.fit(X_train, y_train)

    #         y_pred = clf.predict(X_test)

    #         acc = accuracy_score(y_test, y_pred)
    #         scores.append(acc)
        
    #     return scores

    # def strat_kfold(self, X, y, n_splits = 5):
    #     kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state = 3)

    #     scores = []
        
    #     for i, (train_index, test_index) in enumerate(kf.split(X, y)):
    #         # print("i: {}, (train_index, test_index) = ({}, {})".format(i, train_index, test_index))
    #         X_train = [X[i] for i in train_index].copy()
    #         y_train = [y[i] for i in train_index].copy()
    #         X_test = [X[i] for i in test_index].copy()
    #         y_test = [y[i] for i in test_index].copy()

    #         X_train = pd.concat(X_train)
    #         y_train = pd.concat(y_train)
    #         X_test = pd.concat(X_test)
    #         y_test = pd.concat(y_test)

    #         if self.scaler:
    #             new_scaler = clone(self.scaler)
    #             X_train = new_scaler.fit_transform(X_train)
    #             X_test = new_scaler.transform(X_test)
    #         if self.pca:
    #             new_pca = clone(self.pca)
    #             X_train = new_pca.fit_transform(X_train)
    #             X_test = new_pca.transform(X_test)

    #         clf = clone(self.clf)

    #         clf.fit(X_train, y_train)

    #         y_pred = clf.predict(X_test)

    #         acc = accuracy_score(y_test, y_pred)
    #         scores.append(acc)
        
    #     return scores


    # def visualize_decision_bounds(self):
    #     for i in range(len(self.features)):
    #         for j in range(i+1, len(self.features)):
    #             graph_decision_boundaries(self.X_train[:, [i, j]], self.y_train, self.clf, feature_names=[self.features[i], self.features[j]])
    
    # def get_gini_importance(self):
    #     if hasattr(self.clf, "n_features_in_"):
    #         imp = self.clf.feature_importances_
    #         gini_frame = pd.DataFrame({"Features" : self.features, "Gini Imp" : imp})

    #         return(gini_frame)
    #     else:
    #         print("Classifier hasn't been trained")
    
    # def get_permutation_importance(self, X, y):
    #     if hasattr(self.clf, "n_features_in_"):
    #         X_test = X.copy()

    #         base_acc, _ = self.test(X_test, y)

    #         imp = []
    #         for feat in self.features:
    #             X_shuff = X_test.copy()
    #             np.random.shuffle(X_shuff[feat].values)
    #             shuff_acc, _ = self.test(X_shuff, y)
    #             imp.append(base_acc - shuff_acc)
            
    #         perm_frame = pd.DataFrame({"Features" : self.features, "Perm Imp" : imp})

    #         return(perm_frame)
                
    #     else:
    #         print("Classifier hasn't been trained")
    
    # def get_drop_importance(self, X, y):
    #     if hasattr(self.clf, "n_features_in_"):
    #         X_test = X.copy()

    #         base_acc, _ = self.test(X_test, y)

    #         drop_acc = []
    #         for i in self.features:
    #             drop_features = self.features.copy()
    #             drop_features.remove(i)
    #             X_train_drop = self.X_train.copy()[drop_features]
    #             y_train_drop = self.y_train.copy()
    #             X_test_drop = X_test.copy()[drop_features]
    #             y_test_drop = y.copy()

    #             if self.scaler:
    #                 scaler = StandardScaler()
    #                 X_train_drop = scaler.fit_transform(X_train_drop)
    #                 X_test_drop = scaler.transform(X_test_drop)
    #             if self.pca:
    #                 pca = PCA(n_components=self.n_components)
    #                 X_train_drop = pca.fit_transform(X_train_drop)
    #                 X_test_drop = pca.transform(X_test_drop)

    #             drop_clf = clone(self.clf)
    #             drop_clf.fit(X_train_drop, y_train_drop)

    #             y_pred = drop_clf.predict(X_test_drop)
    #             acc = accuracy_score(y_test_drop, y_pred)
                
    #             drop_acc.append(base_acc - acc)
            
    #         drop_frame = pd.DataFrame({"Features" : self.features, "Drop Imp" : drop_acc})

    #         return(drop_frame)

    #     else:
    #         print("Classifier hasn't been trained")
        
    # def get_importance_table(self, X, y):
    #     X_test = X.copy()
    #     y_test = y.copy()

    #     gini = self.get_gini_importance().set_index('Features')
    #     perm = self.get_permutation_importance(X_test, y_test).set_index('Features')
    #     drop = self.get_drop_importance(X_test, y_test).set_index('Features')

    #     imp_frame = pd.concat([gini, perm, drop], axis=1)

    #     return imp_frame




