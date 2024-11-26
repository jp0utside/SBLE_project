import numpy as np
import pandas as pd
from random_forest import *
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.base import clone

def analyze_rf(model, data, cv = None):
    preds, true, split, acc = get_rf_preds(model, data, cv)
    majority_preds = []
    for pred in preds:
        majority = get_cumulative_majority(pred)
        majority_preds.append(majority)

    raw_acc = []
    maj_acc = []
    raw_cmats = []
    maj_cmats = []
    first_correct = []
    last_incorrect = []

    # Analyze accuracy for each trip with raw predictions
    for idx in range(len(preds)):
        if len(true[idx] > 0):
            raw_acc.append(accuracy_score(true[idx], preds[idx]))
            maj_acc.append(accuracy_score(true[idx], majority_preds[idx]))
            raw_cmats.append(confusion_matrix(true[idx], preds[idx]))
            maj_cmats.append(confusion_matrix(true[idx], majority_preds[idx]))
            correct = preds[idx] == true[idx]
        
            i = 0
            while correct[i] != 1:
                i += 1

            first_correct.append(i)

            i = len(correct) - 1
            while (correct[i] != 0 and i > 0):
                i -= 1
            
            last_incorrect.append(i)
        else:
            raw_acc.append(-1)
            maj_acc.append(-1)
            raw_cmats.append(-1)
            maj_cmats.append(-1)
            first_correct.append(-1)
            last_incorrect.append(-1)
    
    return raw_acc, raw_cmats, maj_acc, maj_cmats, first_correct, last_incorrect

def analyze_mlp(model, data, cv = None):
    preds, true, split, acc = get_mlp_preds(model, data, cv)
    majority_preds = []
    for pred in preds:
        majority = get_cumulative_majority(pred)
        majority_preds.append(majority)

    raw_acc = []
    maj_acc = []
    raw_cmats = []
    maj_cmats = []
    first_correct = []
    last_incorrect = []

    # Analyze accuracy for each trip with raw predictions
    for idx in range(len(preds)):
        if len(true[idx] > 0):
            raw_acc.append(accuracy_score(true[idx], preds[idx]))
            maj_acc.append(accuracy_score(true[idx], majority_preds[idx]))
            raw_cmats.append(confusion_matrix(true[idx], preds[idx]))
            maj_cmats.append(confusion_matrix(true[idx], majority_preds[idx]))
            correct = preds[idx] == true[idx]
        
            i = 0
            while correct[i] != 1:
                i += 1

            first_correct.append(i)

            i = len(correct) - 1
            while (correct[i] != 0 and i > 0):
                i -= 1
            
            last_incorrect.append(i)
        else:
            raw_acc.append(-1)
            raw_cmats.append(-1)
            maj_acc.append(-1)
            maj_cmats.append(-1)
            first_correct.append(-1)
            last_incorrect.append(-1)
    
    return raw_acc, raw_cmats, maj_acc, maj_cmats, first_correct, last_incorrect
    
# Expects concatenated data
def get_rf_preds(model, data, cv = None):
    X = data[model.features]
    y = data["seat"]
    if cv is None:
        cv = StratifiedGroupKFold(n_splits = 5)
    num_groups = max(data["group"]) + 1

    preds = [pd.DataFrame() for i in range(num_groups)]
    true = [pd.DataFrame() for i in range(num_groups)]
    split = [-1 for i in range(num_groups)]
    accs = [-1 for i in range(cv.n_splits)]
    groups = list(data["group"])

    for i, (train_index, test_index) in enumerate(cv.split(X, y, groups)):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]


        temp_model = clone(model)

        temp_model.fit(X_train, y_train)

        output = temp_model.predict(X_test)

        test_groups = list(set(list(test_data["group"])))

        for group in test_groups:
            group_mask = test_data["group"] == group
            group_preds = output[group_mask]
            preds[group] = group_preds
            group_true = y_test.loc[group_mask]
            true[group] = group_true.to_numpy()
            split[group] = i
        
        acc = accuracy_score(y_test, output)
        accs[i] = acc
    return preds, true, split, accs
    
# Expects concatenated data
def get_mlp_preds(model, data, cv = None):
    X = data[model.features]
    y = data["seat"]
    if cv is None:
        cv = StratifiedGroupKFold(n_splits = 5)
    num_groups = max(data["group"]) + 1

    preds = [pd.DataFrame() for i in range(num_groups)]
    true = [pd.DataFrame() for i in range(num_groups)]
    split = [-1 for i in range(num_groups)]
    accs = [-1 for i in range(cv.n_splits)]
    groups = list(data["group"])

    for i, (train_index, test_index) in enumerate(cv.split(X, y, groups)):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]


        temp_model = clone(model)

        temp_model.fit(X_train, y_train)

        output = temp_model.predict(X_test)

        test_groups = list(set(list(test_data["group"])))

        for group in test_groups:
            group_mask = test_data["group"] == group
            group_preds = output[group_mask]
            preds[group] = group_preds
            group_true = y_test.loc[group_mask]
            true[group] = group_true.to_numpy()
            split[group] = i
        
        acc = accuracy_score(y_test, output)
        accs[i] = acc
    return preds, true, split, accs

# Expects array of data frames
def get_lstm_preds(model, data, cv = None):
    X = [frame[["rssi_1", "rssi_accuracy_1", "rssi_2", "rssi_accuracy_2"]] for frame in data]
    y = [frame.iloc[0]["seat"] if frame.shape[0] > 0 else -1 for frame in data]

    if cv is None:
        cv = StratifiedKFold(n_splits=5)

    num_groups = max([frame.iloc[0]["group"] if frame.shape[0] > 0 else -1 for frame in data]) + 1

    preds = [pd.DataFrame() for i in range(num_groups)]
    true = [pd.DataFrame() for i in range(num_groups)]
    split = [-1 for i in range(num_groups)]
    accs = [-1 for i in range(cv.n_splits)]
    groups = list(data["group"])

    for i, (train_index, test_index) in enumerate(cv.split(X, y)):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]
        X_train = [X[idx] for idx in train_index]
        X_test = [X[idx] for idx in test_index]
        y_train = [y[idx] for idx in train_index]
        y_test = [y[idx] for idx in test_index]

        temp_model = clone(model)
        temp_model.fit(X_train, y_train)

        output, seq_idxs = temp_model.predict(X_test)
        y_true = [y_test[idx] for idx in seq_idxs]

        for idx in set(seq_idxs):
            group_mask = seq_idxs == idx
            trip_idx = test_index[idx]
            preds[trip_idx] = output[group_mask]
            true[trip_idx] = [y_test[idx]]*len(preds[trip_idx])
            split[trip_idx] = i
        
        acc = accuracy_score(y_true, output)
        accs[i] = acc
    return preds, true, split, accs


def get_cumulative_majority(preds):
    majority = [0 for i in range(len(preds))]
    current = {0:0, 1:0, 2:0}

    for idx, pred in enumerate(preds):
        current[pred] += 1
        majority[idx] = max(zip(current.values(), current.keys()))[1]

    return majority


    

        


