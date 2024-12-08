import numpy as np
import pandas as pd
from random_forest import *
from mlp import *
from lstm import *
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.base import clone

def full_suite(rf, mlp, lstm):
    """
    Metrics:
        - Raw Accuray
        - F1 score (or something)
        - Weighted f1 score
        - Some way to measure how concentrated misclassified points are
            (i.e. are the errors all over or are certain trips just always misclassified?)
        - Maybe that's the majority voting metric. 
        - Confusion Matrices
        - 
    """

    rf_data = analyze_rf(rf)
    mlp_data = analyze_mlp(mlp)
    lstm_data = analyze_lstm(lstm)

    return rf_data, mlp_data, lstm_data


def analyze_rf(model, data, cv = None):
    preds, true, split, acc = get_rf_preds(model, data, cv)

    data = {'acc': [], 'f1': [], 'cmat': [], 'split_acc': [], 'split_f1': [],
            'split_cmats': [], 'trip_acc': [], 'trip_f1': [], 'trip_cmats': []}

    # Analyze accuracy for each trip with raw predictions
    for idx in range(len(preds)):
        if len(true[idx] > 0):
            data['trip_acc'].append(accuracy_score(true[idx], preds[idx]))
            data['trip_cmats'].append(confusion_matrix(true[idx], preds[idx]))
            data['f1_scores'].append(f1_score(true[idx], preds[idx], average = 'macro'))
        else:
            data['trip_acc'].append(-1)
            data['trip_cmats'].append(-1)

    for i in range(max(split) + 1):
        accs = [data['trip_acc'][idx] for idx in range(len(data['trip_acc'])) if split[idx] == i]
        f1s = [data['trip_f1'][idx] for idx in range(len(data['trip_f1'])) if split[idx] == i]
        cms = [data['trip_cmats'][idx] for idx in range(len(data['trip_cmats'])) if split[idx] == i]

        data['split_acc'].append(np.mean(accs))
        data['split_f1'].append(np.mean(f1s))
        data.append(np.sum(cms, axis = 0))

    data['acc'] = [np.mean(data['split_acc'])]
    data['f1'] = [np.mean(data['split_f1'])]
    data['cmat'] = [np.sum(data['split_cmats'])]
    
    return data

def analyze_mlp(model, data, cv = None):
    preds, true, split, acc = get_mlp_preds(model, data, cv)

    data = {'acc': [], 'f1': [], 'cmat': [], 'split_acc': [], 'split_f1': [],
            'split_cmats': [], 'trip_acc': [], 'trip_f1': [], 'trip_cmats': []}

    # Analyze accuracy for each trip with raw predictions
    for idx in range(len(preds)):
        if len(true[idx] > 0):
            data['trip_acc'].append(accuracy_score(true[idx], preds[idx]))
            data['trip_cmats'].append(confusion_matrix(true[idx], preds[idx]))
            data['f1_scores'].append(f1_score(true[idx], preds[idx], average = 'macro'))
        else:
            data['trip_acc'].append(-1)
            data['trip_cmats'].append(-1)

    for i in range(max(split) + 1):
        accs = [data['trip_acc'][idx] for idx in range(len(data['trip_acc'])) if split[idx] == i]
        f1s = [data['trip_f1'][idx] for idx in range(len(data['trip_f1'])) if split[idx] == i]
        cms = [data['trip_cmats'][idx] for idx in range(len(data['trip_cmats'])) if split[idx] == i]

        data['split_acc'].append(np.mean(accs))
        data['split_f1'].append(np.mean(f1s))
        data.append(np.sum(cms, axis = 0))

    data['acc'] = [np.mean(data['split_acc'])]
    data['f1'] = [np.mean(data['split_f1'])]
    data['cmat'] = [np.sum(data['split_cmats'])]
    
    return data

def analyze_lstm(model, data, cv = None):
    preds, true, split, acc = get_rf_preds(model, data, cv)

    # Analyze accuracy for each trip with raw predictions
    for idx in range(len(preds)):
        if len(true[idx] > 0):
            data['trip_acc'].append(accuracy_score(true[idx], preds[idx]))
            data['trip_cmats'].append(confusion_matrix(true[idx], preds[idx]))
            data['f1_scores'].append(f1_score(true[idx], preds[idx], average = 'macro'))
        else:
            data['trip_acc'].append(-1)
            data['trip_cmats'].append(-1)


    for i in range(max(split) + 1):
        accs = [data['trip_acc'][idx] for idx in range(len(data['trip_acc'])) if split[idx] == i]
        f1s = [data['trip_f1'][idx] for idx in range(len(data['trip_f1'])) if split[idx] == i]
        cms = [data['trip_cmats'][idx] for idx in range(len(data['trip_cmats'])) if split[idx] == i]

        data['split_acc'].append(np.mean(accs))
        data['split_f1'].append(np.mean(f1s))
        data.append(np.sum(cms, axis = 0))
    
    data['acc'] = [np.mean(data['split_acc'])]
    data['f1'] = [np.mean(data['split_f1'])]
    data['cmat'] = [np.sum(data['split_cmats'])]

    return data

    
# Expects concatenated data
def get_rf_preds(model, data, cv = None):
    X = [frame[["rssi_1", "rssi_accuracy_1", "rssi_2", "rssi_accuracy_2"]] for frame in data]
    y = [frame.iloc[0]["seat"] if frame.shape[0] > 0 else -1 for frame in data]

    if cv is None:
        cv = StratifiedKFold(n_splits = 5)

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
    X = [frame[["rssi_1", "rssi_accuracy_1", "rssi_2", "rssi_accuracy_2"]] for frame in data]
    y = [frame.iloc[0]["seat"] if frame.shape[0] > 0 else -1 for frame in data]
    
    if cv is None:
        cv = StratifiedKFold(n_splits = 5)

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



    

        


