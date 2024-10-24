import numpy as np
import pandas as pd
from visualize import graph_decision_boundaries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, confusion_matrix

class RandomForest:
    def __init__(self, n_estimators=100, random_state=3, use_scaler = True, use_pca = True, n_components=2):
        if use_scaler:
            self.scaler = StandardScaler()
        else:
            self.scaler = None
        if use_pca:
            self.pca = PCA(n_components=n_components)
            self.n_components = n_components
        else:
            self.pca = None
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        
    def train(self, X, y):
        X_train = X.copy()
        self.X_train = X_train.copy()
        self.y_train = y.copy()

        if self.scaler:
            X_train = self.scaler.fit_transform(X_train)
        if self.pca:
            X_train = self.pca.fit_transform(X_train)

        self.clf.fit(X_train, y)

        self.features = []
        if self.pca:
            for i in range(X_train.shape[1]):
                self.features.append("Feature {}".format(i))
        else:
            for i in X.columns:
                self.features.append(i)

    
    def test(self, X, y):
        X_test = X.copy()
        if self.scaler:
            X_test = self.scaler.transform(X_test)
        if self.pca:
            X_test = self.pca.transform(X_test)

        y_pred = self.clf.predict(X_test)

        acc = accuracy_score(y, y_pred)

        conf_matrix = confusion_matrix(y, y_pred)
        cm = pd.DataFrame(conf_matrix, ["front", "middle", "back"], columns = ["Predicted front", "Predicted middle", "Predicted back"])

        # print("Accuracy score: " + str(acc))
        # print(cm)
        return acc, cm
    
    def visualize_decision_bounds(self):
        for i in range(len(self.features)):
            for j in range(i+1, len(self.features)):
                graph_decision_boundaries(self.X_train[:, [i, j]], self.y_train, self.clf, feature_names=[self.features[i], self.features[j]])
    
    def get_gini_importance(self):
        if hasattr(self.clf, "n_features_in_"):
            imp = self.clf.feature_importances_
            gini_frame = pd.DataFrame({"Features" : self.features, "Gini Imp" : imp})

            return(gini_frame)
        else:
            print("Classifier hasn't been trained")
    
    def get_permutation_importance(self, X, y):
        if hasattr(self.clf, "n_features_in_"):
            X_test = X.copy()

            base_acc, _ = self.test(X_test, y)

            imp = []
            for feat in self.features:
                X_shuff = X_test.copy()
                np.random.shuffle(X_shuff[feat].values)
                shuff_acc, _ = self.test(X_shuff, y)
                imp.append(base_acc - shuff_acc)
            
            perm_frame = pd.DataFrame({"Features" : self.features, "Perm Imp" : imp})

            return(perm_frame)
                
        else:
            print("Classifier hasn't been trained")
    
    def get_drop_importance(self, X, y):
        if hasattr(self.clf, "n_features_in_"):
            X_test = X.copy()

            base_acc, _ = self.test(X_test, y)

            drop_acc = []
            for i in self.features:
                drop_features = self.features.copy()
                drop_features.remove(i)
                X_train_drop = self.X_train.copy()[drop_features]
                y_train_drop = self.y_train.copy()
                X_test_drop = X_test.copy()[drop_features]
                y_test_drop = y.copy()

                if self.scaler:
                    scaler = StandardScaler()
                    X_train_drop = scaler.fit_transform(X_train_drop)
                    X_test_drop = scaler.transform(X_test_drop)
                if self.pca:
                    pca = PCA(n_components=self.n_components)
                    X_train_drop = pca.fit_transform(X_train_drop)
                    X_test_drop = pca.transform(X_test_drop)

                drop_clf = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
                drop_clf.fit(X_train_drop, y_train_drop)

                y_pred = drop_clf.predict(X_test_drop)
                acc = accuracy_score(y_test_drop, y_pred)
                
                drop_acc.append(base_acc - acc)
            
            drop_frame = pd.DataFrame({"Features" : self.features, "Drop Imp" : drop_acc})

            return(drop_frame)

        else:
            print("Classifier hasn't been trained")
        
    def get_importance_table(self, X, y):
        X_test = X.copy()
        y_test = y.copy()

        gini = self.get_gini_importance().set_index('Features')
        perm = self.get_permutation_importance(X_test, y_test).set_index('Features')
        drop = self.get_drop_importance(X_test, y_test).set_index('Features')

        imp_frame = pd.concat([gini, perm, drop], axis=1)

        return imp_frame




