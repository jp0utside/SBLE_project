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
    def __init__(self, n_components=2, n_estimators=100, random_state=3):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        
    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        X_reduced = self.pca.fit_transform(X_scaled)
        self.clf.fit(X_reduced, y)
        self.X_train = X_reduced
        self.y_train = y
        self.features = []
        for i in range(X_reduced.shape[1]):
            self.features.append("PCA {}".format(i))
    
    def test(self, X, y):
        X_test = self.scaler.transform(X)
        X_test = self.pca.transform(X_test)

        y_pred = self.clf.predict(X_test)

        acc = accuracy_score(y, y_pred)

        conf_matrix = confusion_matrix(y, y_pred)
        cm = pd.DataFrame(conf_matrix, ["front", "middle", "back"], columns = ["Predicted front", "Predicted middle", "Predicted back"])

        print("Accuracy score: " + str(acc))
        print(cm)
    
    def visualize_decision_bounds(self):
        for i in range(len(self.features)):
            for j in range(i+1, len(self.features)):
                graph_decision_boundaries(self.X_train[:, [i, j]], self.y_train, self.clf, feature_names=[self.features[i], self.features[j]])