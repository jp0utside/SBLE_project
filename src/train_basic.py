import numpy as np
from parse import *
from filter import *
from visualize import *
import trip
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, confusion_matrix

"""
Function to evaluate performance of classifier.
"""
def evaluate_clf(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)

    conf_matrix = confusion_matrix(y_test, y_pred)

    cm = pd.DataFrame(conf_matrix, ["front", "middle", "back"], columns = ["Predicted front", "Predicted middle", "Predicted back"])

    print("Accuracy score: " + str(acc))
    print(cm)

"""
PCA wrapper to scale data and return pca, scaler, and reduced dataset
"""
def get_pca(X, n_components = 2):
    scaler = StandardScaler()
    pca = PCA(n_components=n_components)

    X_scaled = scaler.fit_transform(X)
    X_reduced = pca.fit_transform(X_scaled)

    return pca, scaler, X_reduced

"""
Basic linear seperator trained on average RSSI difference values for each trip.
Attempts to generate highest true positive rate for each seat.
Replicate's the initial work done in Matlab but with Python data.
"""
def basic_linear(data):
    new_data = get_avg_rssi_diff(data)
    front = new_data.loc[new_data["seat"] == "front", "rssi"].tolist()
    middle = new_data.loc[new_data["seat"] == "middle", "rssi"].tolist()
    back = new_data.loc[new_data["seat"] == "back", "rssi"].tolist()

    totDiff = [front, middle, back]
    totLen = sum([len(totDiff[0]), len(totDiff[1]), len(totDiff[2])])
    pos_inf = float('inf')
    neg_inf = float('-inf')

    best_left = neg_inf
    best_right = pos_inf
    best_matrix = []

    min_acc = neg_inf

    left_range = list(range(-15, 0))
    right_range = list(range(0,15))

    for left in left_range:
        for right in right_range:
            rng = [neg_inf, left, right, pos_inf]
            a = [[[0] for x in range(3)] for y in range(3)]
            acc = 0
            for i in range(3):
                for j in range(3):
                    a[i][j] = sum([totDiff[i][x] > rng[j] and totDiff[i][x] < rng[j+1] for x in range(len(totDiff[i]))])/len(totDiff[i])
                acc += (a[i][i]*(len(totDiff[i])/totLen))
            if acc > min_acc:
                best_left = left
                best_right = right
                min_acc = acc
                best_matrix = a
    
    rand_acc = (len(totDiff[0])/totLen)**2 + (len(totDiff[1])/totLen)**2 + (len(totDiff[2])/totLen)**2

    print("Random accuracy: " + str(rand_acc))
    print("Best accuracy: " + str(min_acc))
    print("Best range: " + str([best_left, best_right]))
    for i in best_matrix:
        print(i)

"""
Trains linear classifier on ALL data points using SGD through Scikit-learn.
"""
def linear_classifier_all_datapoints(data, features = ["rssi_diff"], random_state = 42):
    X_train, X_test, y_train, y_test = train_test_split(data[features], data["seat"].tolist(), test_size = 0.2, random_state=random_state)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = SGDClassifier(random_state=random_state)
    clf.fit(X_train, y_train)

    class_lines = pd.DataFrame(
        np.append(clf.coef_.T, [clf.intercept_], axis = 0),
        features + ["intercept"],
        columns = ["front", "middle", "back"]
    )

    evaluate_clf(clf, X_test, y_test)

"""
Trains linear classifier on average rssi difference for each trip.
"""
def linear_classifier_averages(data, random_state = 42):
    new_data = get_avg_rssi_diff(data)
    X_train, X_test, y_train, y_test = train_test_split(new_data[["rssi"]], new_data["seat"].tolist(), test_size = 0.2, random_state=random_state)

    print(X_train)
    print(X_test)
    print(y_train)
    print(y_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = SGDClassifier(random_state=random_state)
    clf.fit(X_train, y_train)

    class_lines = pd.DataFrame(
        np.append(clf.coef_.T, [clf.intercept_], axis = 0),
        ["rssi", "intercept"],
        columns = ["front", "middle", "back"]
    )

    evaluate_clf(clf, X_test, y_test)

"""
Wrapper function to train and return a simple gaussian mixture model using inputted features.
"""

def gaussian_mixture_model(data, features = ["rssi_1", "rssi_2"], n_comp = 1):
    X = data[features]
    y = data[["seat"]]

    gmm = GaussianMixture(n_components = n_comp)
    gmm.fit(X)

    print("Means:\n", gmm.means_)
    print("Covariances:\n", gmm.covariances_)
    print("Weights:\n", gmm.weights_)

    return gmm

"""
Function to measure correlation coefficients between features using pearson coefficient.
Can visualize if desired.
"""

def measure_correlation(X, features, visualize = False, top_x = 10):
    correlations = {}
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            coef = np.corrcoef(X[features[i]].to_list(), X[features[j]].to_list())[0,1]
            correlations[(features[i], features[j])] = coef
    
    correlations = dict(sorted(correlations.items(), key = lambda item: abs(item[1]), reverse = True))

    if visualize:
        for i in range(top_x):
            key = list(correlations.keys())[i]
            x = X[key[0]]
            y = X[key[1]]

            graph_correlation(x, y, correlations[key], xlabel=key[0], ylabel=key[1])

    return correlations






    
