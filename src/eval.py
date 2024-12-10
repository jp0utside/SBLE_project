import numpy as np
import pandas as pd
from random_forest import *
from mlp import *
from lstm import *
from parse import *
from filter import *
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.base import clone
from concurrent.futures import ThreadPoolExecutor

all_features = ['rssi_1', 'rssi_accuracy_1', 'rssi_2', 'rssi_accuracy_2', 'weekminute', 'latitude',
       'longitude', 'speed', 'speedAcc', 'vertical_acc', 'altitude', 'course',
       'courseAcc', 'heading', 'horizontal_acc', 'attitude_pitch',
       'attitude_roll', 'attitude_yaw', 'rotation_rate_x', 'rotation_rate_y',
       'rotation_rate_z', 'gravity_accel_x', 'gravity_accel_y',
       'gravity_accel_z', 'user_accel_x', 'user_accel_y', 'user_accel_z',
       'magnetic_field_x', 'magnetic_field_y', 'magnetic_field_z']
    
rssi_features = ['rssi_1', 'rssi_accuracy_1', 'rssi_2', 'rssi_accuracy_2']
position_features = ['latitude',
       'longitude', 'speed', 'speedAcc', 'vertical_acc', 'altitude', 'course',
       'courseAcc', 'heading', 'horizontal_acc', 'attitude_pitch',
       'attitude_roll', 'attitude_yaw', 'rotation_rate_x', 'rotation_rate_y',
       'rotation_rate_z', 'gravity_accel_x', 'gravity_accel_y',
       'gravity_accel_z', 'user_accel_x', 'user_accel_y', 'user_accel_z',
       'magnetic_field_x', 'magnetic_field_y', 'magnetic_field_z']

def load_eval(rf_fp = 'rf_data_temp.pickle', mlp_fp = 'mlp_data_temp.pickle', lstm_fp = 'lstm_data_temp.pickle'):
    with open(rf_fp, 'rb') as f:
        rf_data = pickle.load(f)

    with open(mlp_fp, 'rb') as f:
        mlp_data = pickle.load(f)
    
    with open(lstm_fp, 'rb') as f:
        lstm_data = pickle.load(f)
    
    return rf_data, mlp_data, lstm_data


def full_suite(rf, mlp, lstm, trips = [], split_seed = 33):
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
    if not trips:
        trips = get_trips_quick()
    data = get_tagged_dataset(trips)

    if lstm is not None:
        data = [frame for frame in data if frame.shape[0] >= lstm.sub_sequence_length]
    
    X = [frame[all_features] for frame in data]
    y = [frame.iloc[0]["seat"] if frame.shape[0] > 0 else -1 for frame in data]

    cv = StratifiedKFold(n_splits = 5, random_state = split_seed, shuffle = True)

    def run_rf():
        return analyze_rf(rf, X, y, cv) if rf is not None else None
    
    def run_mlp():
        return analyze_mlp(mlp, X, y, cv) if mlp is not None else None
    
    def run_lstm():
        return analyze_lstm(lstm, X, y, cv) if lstm is not None else None
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        rf_future = executor.submit(run_rf)
        mlp_future = executor.submit(run_mlp)
        lstm_future = executor.submit(run_lstm)

        rf_results = rf_future.result()
        mlp_results = mlp_future.result()
        lstm_results = lstm_future.result()


    return rf_results, mlp_results, lstm_results

def analyze_rf(model, X, y, cv = None):
    preds, true, split = get_rf_preds(model, X, y, cv)

    data = {'acc': [], 'f1': [], 'cmat': [], 'split_acc': [], 'split_f1': [],
            'split_cmats': [], 'trip_acc': [], 'trip_f1': [], 'trip_cmats': []}

    # Analyze accuracy for each trip with raw predictions
    for idx, (p, t) in enumerate(zip(preds, true)):
        if len(t) > 0 and t[0] != -1:
            data['trip_acc'].append(accuracy_score(t, p))
            data['trip_f1'].append(f1_score(t, p, average = 'macro'))
            data['trip_cmats'].append(confusion_matrix(t, p, labels = [0, 1, 2]))
        else:
            data['trip_acc'].append(-1)
            data['trip_f1'].append(-1)
            data['trip_cmats'].append([])

    for i in range(max(split) + 1):
        split_preds = []
        split_true = []
        for tidx, (p, t) in enumerate(zip(preds, true)):
            if split[tidx] == i and t[0] != -1:
                split_preds.extend(p)
                split_true.extend(t)

        data['split_acc'].append(accuracy_score(split_true, split_preds))
        data['split_f1'].append(f1_score(split_true, split_preds, average='macro'))
        data['split_cmats'].append(confusion_matrix(split_true, split_preds, labels = [0,1,2]))

    data['acc'] = [np.mean(data['split_acc'])]
    data['f1'] = [np.mean(data['split_f1'])]
    data['cmat'] = [np.sum(data['split_cmats'], axis = 0)]

    print("RF")
    print("split_acc: ", data['split_acc'])
    print("split_f1: ", data['split_f1'])
    print("acc: ", data['acc'])
    print("f1: ", data['f1'])
    print("cmat: ", data['cmat'])
    print()
    
    return data

def analyze_mlp(model, X, y, cv = None):
    preds, true, split = get_mlp_preds(model, X, y, cv)

    data = {'acc': [], 'f1': [], 'cmat': [], 'split_acc': [], 'split_f1': [],
            'split_cmats': [], 'trip_acc': [], 'trip_f1': [], 'trip_cmats': []}

    # Analyze accuracy for each trip with raw predictions
    for idx, (p, t) in enumerate(zip(preds, true)):
        if len(t) > 0 and t[0] != -1:
            data['trip_acc'].append(accuracy_score(t, p))
            data['trip_f1'].append(f1_score(t, p, average = 'macro'))
            data['trip_cmats'].append(confusion_matrix(t, p, labels = [0, 1, 2]))
        else:
            data['trip_acc'].append(-1)
            data['trip_f1'].append(-1)
            data['trip_cmats'].append([])

    for i in range(max(split) + 1):
        split_preds = []
        split_true = []

        for tidx, (p, t) in enumerate(zip(preds, true)):
            if split[tidx] == i and t[0] != -1:
                split_preds.extend(p)
                split_true.extend(t)

        # for tidx in range(len(preds)):
        #     if split[tidx] == i and true[tidx][0] != -1:
        #         split_preds.extend(preds[tidx])
        #         split_true.extend(true[tidx])

        data['split_acc'].append(accuracy_score(split_true, split_preds))
        data['split_f1'].append(f1_score(split_true, split_preds, average='macro'))
        data['split_cmats'].append(confusion_matrix(split_true, split_preds, labels = [0,1,2]))

    data['acc'] = [np.mean(data['split_acc'])]
    data['f1'] = [np.mean(data['split_f1'])]
    data['cmat'] = [np.sum(data['split_cmats'], axis = 0)]

    print("MLP")
    print("split_acc: ", data['split_acc'])
    print("split_f1: ", data['split_f1'])
    print("acc: ", data['acc'])
    print("f1: ", data['f1'])
    print("cmat: ", data['cmat'])
    print()
    
    return data

def analyze_lstm(model, X, y, cv = None):
    preds, true, split = get_lstm_preds(model, X, y, cv)

    data = {'acc': [], 'f1': [], 'cmat': [], 'split_acc': [], 'split_f1': [],
            'split_cmats': [], 'trip_acc': [], 'trip_f1': [], 'trip_cmats': []}

    # Analyze accuracy for each trip with raw predictions
    for idx, (p, t) in enumerate(zip(preds, true)):
        if len(t) > 0 and t[0] != -1:
            data['trip_acc'].append(accuracy_score(t, p))
            data['trip_f1'].append(f1_score(t, p, average = 'macro'))
            data['trip_cmats'].append(confusion_matrix(t, p, labels = [0, 1, 2]))
        else:
            data['trip_acc'].append(-1)
            data['trip_f1'].append(-1)
            data['trip_cmats'].append([])

    for i in range(max(split) + 1):
        split_preds = []
        split_true = []
        for tidx, (p, t) in enumerate(zip(preds, true)):
            if split[tidx] == i and t[0] != -1:
                split_preds.extend(p)
                split_true.extend(t)

        data['split_acc'].append(accuracy_score(split_true, split_preds))
        data['split_f1'].append(f1_score(split_true, split_preds, average='macro'))
        data['split_cmats'].append(confusion_matrix(split_true, split_preds, labels = [0,1,2]))
    
    data['acc'] = [np.mean(data['split_acc'])]
    data['f1'] = [np.mean(data['split_f1'])]
    data['cmat'] = [np.sum(data['split_cmats'], axis = 0)]

    print("LSTM")
    print("split_acc: ", data['split_acc'])
    print("split_f1: ", data['split_f1'])
    print("acc: ", data['acc'])
    print("f1: ", data['f1'])
    print("cmat: ", data['cmat'])
    print()

    return data

    
def get_rf_preds(model, X, y, cv = None):

    if cv is None:
        cv = StratifiedKFold(n_splits = 5)

    preds = [pd.DataFrame() for i in range(len(X))]
    true = [pd.DataFrame() for i in range(len(X))]
    split = [-1 for i in range(len(X))]

    for i, (train_index, test_index) in enumerate(cv.split(X, y)):
        X_train = [X[idx] for idx in train_index]
        y_train = [y[idx] for idx in train_index]

        temp_model = clone(model)

        temp_model.fit(X_train, y_train)

        for idx in test_index:
            X_test = [X[idx]]
            y_test = pd.Series([y[idx]]*X[idx].shape[0])
            y_pred = temp_model.predict(X_test)
            preds[idx] = y_pred
            true[idx] = y_test
            split[idx] = i
        
    return preds, true, split
    

def get_mlp_preds(model, X, y, cv = None):
    
    if cv is None:
        cv = StratifiedKFold(n_splits = 5)

    preds = [pd.DataFrame() for i in range(len(X))]
    true = [pd.DataFrame() for i in range(len(X))]
    split = [-1 for i in range(len(X))]

    for i, (train_index, test_index) in enumerate(cv.split(X, y)):
        X_train = [X[idx] for idx in train_index]
        y_train = [y[idx] for idx in train_index]

        temp_model = clone(model)
        temp_model.fit(X_train, y_train)

        for idx in test_index:
            X_test = [X[idx]]
            y_test = pd.Series([y[idx]]*X[idx].shape[0])
            y_pred = temp_model.predict(X_test)
            preds[idx] = y_pred
            true[idx] = y_test
            split[idx] = i
        
    return preds, true, split

# Expects array of data frames
def get_lstm_preds(model, X, y, cv = None):

    if cv is None:
        cv = StratifiedKFold(n_splits=5)

    preds = [pd.DataFrame() for i in range(len(X))]
    true = [pd.DataFrame() for i in range(len(X))]
    split = [-1 for i in range(len(X))]

    for i, (train_index, test_index) in enumerate(cv.split(X, y)):
        X_train = [X[idx] for idx in train_index]
        y_train = [y[idx] for idx in train_index]

        temp_model = clone(model)
        temp_model.fit(X_train, y_train)

        for idx in test_index:
            X_test = [X[idx]]
            if X[idx].shape[0] >= model.sub_sequence_length:
                y_pred, seq_idx = temp_model.predict(X_test)
                y_test = pd.Series([y[idx]]*len(seq_idx))
                preds[idx] = y_pred
                true[idx] = y_test
                split[idx] = i
            else:
                preds[idx] = [-1]
                true[idx] = [-1]
                split[idx] = i
        
    return preds, true, split



    

        


