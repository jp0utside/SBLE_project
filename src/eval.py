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

"""
Function to load saved results of eval function

Args:
    rf_fp: filepath of saved rf data
    mlp_fp: filepath of saved mlp data
    lstm_fp: filepath of saved lstm data
"""
def load_eval(rf_fp = 'rf_data_temp.pickle', mlp_fp = 'mlp_data_temp.pickle', lstm_fp = 'lstm_data_temp.pickle'):
    with open(rf_fp, 'rb') as f:
        rf_data = pickle.load(f)

    with open(mlp_fp, 'rb') as f:
        mlp_data = pickle.load(f)
    
    with open(lstm_fp, 'rb') as f:
        lstm_data = pickle.load(f)
    
    return rf_data, mlp_data, lstm_data


"""
Function to return the full suite of prediction data generated for each model.
Includes accuracy, macro-averaged f1, and confusion matrices for each trip, each split, and average/total of splits

Args:
    rf: Pre-configured Random Forest model to be used for generating predictions
    mlp: Pre-configured MLP model to be used for generating predictions
    lstm: Pre-configured LSTM model to be used for generating predictions
    trips: array of all trip objects
    split_seed: random seed to be used in generating splits

Returns:
    rf: dictionary containing all prediction results for Random Forest
    mlp: dictionary containing all prediction results for MLP
    lstm: dictionary containing all prediction results for LSTM
"""
def full_suite(rf, mlp, lstm, trips = [], split_seed = 33):
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

"""
Function to generate both trip and aggregate metrics for rf predictions

Args:
    model: Random Forest model to be used for predictions
    X: X data following the convention for input data
    y: y data following the convention for input data
    cv: cross-validation object pre-configured to be used in kfold cross-validation

Returns:
    data: dictionary containing all generated metrics
"""
def analyze_rf(model, X, y, cv = None):
    preds, true, split = get_rf_preds(model, X, y, cv)
    majors = get_majority_preds(preds)

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


"""
Function to generate both trip and aggregate metrics for mlp predictions

Args:
    model: MLP model to be used for predictions
    X: X data following the convention for input data
    y: y data following the convention for input data
    cv: cross-validation object pre-configured to be used in kfold cross-validation

Returns:
    data: dictionary containing all generated metrics
"""
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

"""
Function to generate both trip and aggregate metrics for lstm predictions

Args:
    model: LSTM model to be used for predictions
    X: X data following the convention for input data
    y: y data following the convention for input data
    cv: cross-validation object pre-configured to be used in kfold cross-validation

Returns:
    data: dictionary containing all generated metrics
"""
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

"""
Function to get all predictions for given splits using inputted Random Forest model

Args:
    model: Random Forest model to be used in generating predictions
    X: X data to be used in training and testing
    y: y data to be used in training and testing
    cv: Pre-configured cross-validation object to be used in splitting
Returns:
    preds: 2d array containing predictions for each trip
    true: 2d array containing true values for each trip
    split: 1d array indicating which split each trip is a part of
"""
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
    
"""
Function to get all predictions for given splits using inputted MLP model

Args:
    model: MLP model to be used in generating predictions
    X: X data to be used in training and testing
    y: y data to be used in training and testing
    cv: Pre-configured cross-validation object to be used in splitting
Returns:
    preds: 2d array containing predictions for each trip
    true: 2d array containing true values for each trip
    split: 1d array indicating which split each trip is a part of
"""
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

"""
Function to get all predictions for given splits using inputted LSTM model

Args:
    model: LSTM model to be used in generating predictions
    X: X data to be used in training and testing
    y: y data to be used in training and testing
    cv: Pre-configured cross-validation object to be used in splitting
Returns:
    preds: 2d array containing predictions for each trip
    true: 2d array containing true values for each trip
    split: 1d array indicating which split each trip is a part of
"""
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


"""
Function to get cumulative majority by voting with all predictions up to the given point

Args:
    preds: predictions made by the model

Return:
    majority_preds: array of predictions by majority voting from the prior predictions
"""

def get_majority_preds(preds):
    votes = {0: 0, 1: 0, 2: 0}
    majority_preds = []

    for pred in preds:
        votes[pred] += 1
        max_vote = max(votes, key = votes.get)
        majority_preds.append(max_vote)
    
    return majority_preds


    

        


