from parse import *
from filter import *
import trip
from random_forest import *
from mlp import *
from lstm import *
from datetime import datetime
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, GroupKFold, StratifiedGroupKFold, HalvingGridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.base import clone

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
-------------------------------------------------------------------------------------
TESTING FUNCTIONS
Functions to test each model using appropriate train-test splitting methods
Takes in trips object for efficiency, model with desired hyperparameters
Returns accuracy score and confusion matrix
-------------------------------------------------------------------------------------
"""

def test_random_forest(trips = None, model = None):
    if trips is None:
        trips = get_trips_quick()
    
    if model is None:
        model = RandomForest(scaler = StandardScaler(), features=rssi_features, pca = None)
    
    data = get_tagged_dataset(trips)
    
    X_data = [frame[all_features] for frame in data]
    y_data = [frame["seat"] for frame in data]

    y_targets = [frame.iloc[0] for frame in y_data]

    kf = StratifiedKFold(n_splits=2)

    split_gen = kf.split(X_data, y_targets)

    train_index, test_index = next(split_gen)

    X_train = [X_data[idx] for idx in train_index]
    y_train = [y_targets[idx] for idx in train_index]
    X_test = [X_data[idx] for idx in test_index]
    y_test = pd.concat([pd.Series([y_targets[idx]]*X_data[idx].shape[0]) for idx in test_index])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    

    acc, cm = accuracy_score(y_test, y_pred), confusion_matrix(y_test, y_pred)

    # print("acc: ", acc)
    # print(cm)
    
    return acc, cm

def test_mlp(trips = None, model = None):
    if trips is None:
        trips = get_trips_quick()
    if model is None:
        model = MLP(scaler = RobustScaler(), pca = PCA(n_components = 4), features = rssi_features, pca_features = pca_features)
    
    data = get_tagged_dataset(trips, include_pretrip = False, trim_end_zeros = True)
    
    all_features = ['rssi_1', 'rssi_accuracy_1', 'rssi_2', 'rssi_accuracy_2', 'weekminute', 'latitude',
       'longitude', 'speed', 'speedAcc', 'vertical_acc', 'altitude', 'course',
       'courseAcc', 'heading', 'horizontal_acc', 'attitude_pitch',
       'attitude_roll', 'attitude_yaw', 'rotation_rate_x', 'rotation_rate_y',
       'rotation_rate_z', 'gravity_accel_x', 'gravity_accel_y',
       'gravity_accel_z', 'user_accel_x', 'user_accel_y', 'user_accel_z',
       'magnetic_field_x', 'magnetic_field_y', 'magnetic_field_z']
    pca_features = ['latitude',
       'longitude', 'speed', 'speedAcc', 'vertical_acc', 'altitude', 'course',
       'courseAcc', 'heading', 'horizontal_acc', 'attitude_pitch',
       'attitude_roll', 'attitude_yaw', 'rotation_rate_x', 'rotation_rate_y',
       'rotation_rate_z', 'gravity_accel_x', 'gravity_accel_y',
       'gravity_accel_z', 'user_accel_x', 'user_accel_y', 'user_accel_z',
       'magnetic_field_x', 'magnetic_field_y', 'magnetic_field_z']
    rssi_features = ['rssi_1', 'rssi_accuracy_1', 'rssi_2', 'rssi_accuracy_2']

    X_data = [frame[all_features] for frame in data]
    y_data = [frame["seat"] for frame in data]

    y_targets = [frame.iloc[0] for frame in y_data]

    kf = StratifiedKFold(n_splits=2)

    split_gen = kf.split(X_data, y_targets)

    train_index, test_index = next(split_gen)

    X_train = [X_data[idx] for idx in train_index]
    y_train = [y_targets[idx] for idx in train_index]
    X_test = [X_data[idx] for idx in test_index]
    y_test = pd.concat([pd.Series([y_targets[idx]]*X_data[idx].shape[0]) for idx in test_index])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc, cm = accuracy_score(y_test, y_pred), confusion_matrix(y_test, y_pred)

    # print("acc: ", acc)
    # print(cm)
    
    return acc, cm

def test_lstm(trips = None, model = None):
    if trips is None:
        trips = get_trips_quick()

    if model is None:
        model = SklearnLSTMWrapper(all_features, scaler = StandardScaler(), pca_features=all_features, pca = PCA(n_components=3), hidden_size=50, batch_size=10)

    data = get_tagged_dataset(trips, include_pretrip=False, trim_end_zeros=True)

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
    
    X_data = [frame[all_features] for frame in data]
    y_data = [frame.iloc[0]["seat"] if frame.shape[0] > 0 else -1 for frame in data]

    kf = StratifiedKFold(n_splits=2)

    split_gen = kf.split(X_data, y_data)

    train_index, test_index = next(split_gen)

    X_train = [X_data[idx] for idx in train_index]
    X_test = [X_data[idx] for idx in test_index]
    y_train = [y_data[idx] for idx in train_index]
    y_test = [y_data[idx] for idx in test_index]

    model.fit(X_train, y_train)
    y_pred, test_idx = model.predict(X_test)

    y_true = [y_test[idx] for idx in test_idx]

    acc, cm = accuracy_score(y_true, y_pred), confusion_matrix(y_true, y_pred)
    # print("acc: ", acc)
    # print(cm)

    return acc, cm
    # return preds, y_test, seq_idxs

"""
-------------------------------------------------------------------------------------
KFOLD FUNCTIONS
Functions to perform KFold cross-validation tests for each kind of model
Takes in trips object for efficiency, model with desired hyperparameters
Returns accuracy scores for each of the 5 folds, confusion matrices for each
-------------------------------------------------------------------------------------
"""

def kfold_random_forest(trips = None, model = None):
    if trips is None:
        trips = get_trips_quick()
    
    if model is None:
        model = RandomForest(scaler = StandardScaler(), features=rssi_features, pca = None)
    
    data = get_tagged_dataset(trips, trim_end_zeros = True)

    X_data = [frame[all_features] for frame in data]
    y_data = [frame["seat"] for frame in data]

    y_targets = [frame.iloc[0] for frame in y_data]

    kf = StratifiedKFold(n_splits=5)

    scores = []
    confs = []

    for i, (train_index, test_index) in enumerate(kf.split(X_data, y_targets)):
        split_model = clone(model)
        X_train = [X_data[idx] for idx in train_index]
        y_train = [y_targets[idx] for idx in train_index]
        X_test = [X_data[idx] for idx in test_index]
        y_test = pd.concat([pd.Series([y_targets[idx]]*X_data[idx].shape[0]) for idx in test_index])

        split_model.fit(X_train, y_train)
        y_pred = split_model.predict(X_test)

        # y_test = pd.concat(y_test)

        acc, cm = accuracy_score(y_test, y_pred), confusion_matrix(y_test, y_pred)
        scores.append(acc)
        confs.append(cm)
    
    # print("Scores: ", scores)
    # print("Avg: ", sum(scores)/len(scores))
    # print("CMs: ", confs)

    return scores, confs

def kfold_mlp(trips = None, model = None):
    if trips is None:
        trips = get_trips_quick()
    
    if model is None:
        model = MLP(scaler = StandardScaler(), features=rssi_features, pca = None)
    
    data = get_tagged_dataset(trips, trim_end_zeros = True)

    X_data = [frame[all_features] for frame in data]
    y_data = [frame["seat"] for frame in data]

    y_targets = [frame.iloc[0] for frame in y_data]

    kf = StratifiedKFold(n_splits=5)

    scores = []
    cms = []

    for i, (train_index, test_index) in enumerate(kf.split(X_data, y_targets)):
        split_model = clone(model)
        X_train = [X_data[idx] for idx in train_index]
        y_train = [y_targets[idx] for idx in train_index]
        X_test = [X_data[idx] for idx in test_index]
        y_test = pd.concat([pd.Series([y_targets[idx]]*X_data[idx].shape[0]) for idx in test_index])

        split_model.fit(X_train, y_train)
        y_pred = split_model.predict(X_test)

        # y_test = pd.concat(y_test)

        acc, cm = accuracy_score(y_test, y_pred), confusion_matrix(y_test, y_pred)
        scores.append(acc)
        cms.append(cm)
    
    # print("Scores: ", scores)
    # print("Avg: ", sum(scores)/len(scores))
    return scores, cms

def kfold_lstm(trips = None, model = None):
    if trips is None:
        trips = get_trips_quick()
    
    if model is None:
        model = SklearnLSTMWrapper(features=rssi_features, scaler=StandardScaler(), pca=PCA(n_components = 3), pca_features=position_features, hidden_size=50)

    data = get_tagged_dataset(trips, include_pretrip=False, trim_end_zeros=True)

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

    X_data = [frame[all_features] for frame in data]
    y_data = [frame.iloc[0]["seat"] for frame in data]

    kf = StratifiedKFold(n_splits=5)

    scores = []
    cms = []

    for i, (train_index, test_index) in enumerate(kf.split(X_data, y_data)):
        split_model = clone(model)
        X_train = [X_data[idx] for idx in train_index]
        y_train = [y_data[idx] for idx in train_index]
        X_test = [X_data[idx] for idx in test_index]
        y_test = [y_data[idx] for idx in test_index]

        split_model.fit(X_train, y_train)

        y_pred, test_idx = split_model.predict(X_test)

        y_true = [y_test[idx] for idx in test_idx]

        acc, cm = accuracy_score(y_true, y_pred), confusion_matrix(y_true, y_pred)
        scores.append(acc)
        cms.append(cm)

    return scores, cms

"""
-------------------------------------------------------------------------------------
GRIDSEARCH FUNCTIONS
Functions to help perform gridsearches to find optimal hyperparameters for each model
Takes in trips object for efficiency, and desired parameters
-------------------------------------------------------------------------------------
"""

"""
Function to perform gridsearch for random forest model using inputted parameter options
trips: array of trip objects holding the data. trips will be called for all users if no object is used.
params: dictionary of parameter options to be used in gridsearch.
"""
def random_forest_gridsearch(trips = None, params = None, method = 'grid', n_iter = 128):
    if trips is None:
        trips = get_trips_quick()

    data = get_tagged_dataset(trips, trim_end_zeros=True)

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
    
    if params is None:
        params = {
        'features': [rssi_features],
        'scaler': [RobustScaler()],
        'n_estimators': [50, 100],
        'max_depth' : [20],
        'min_samples_split' : [10],
        'min_samples_leaf' : [4],
        'max_features' : ['sqrt'],
    }

    kf = StratifiedKFold(n_splits=5)

    X_data = [frame[all_features] for frame in data]
    y_data = [frame.iloc[0]["seat"] if frame.shape[0] > 0 else -1 for frame in data]

    if method == 'random':
        grid_search = RandomizedSearchCV(RandomForest(), params, n_iter = n_iter, cv = kf, scoring = rf_prediction_scorer, n_jobs = -1, verbose = 5, return_train_score = True)
    elif method == 'halving':
        grid_search = HalvingGridSearchCV(RandomForest(), params, cv = kf, scoring = rf_prediction_scorer, n_jobs = -1, verbose = 5, return_train_score = True)
    else:
        grid_search = GridSearchCV(RandomForest(), params, cv = kf, scoring = rf_prediction_scorer, n_jobs = -1, verbose = 5, return_train_score = True)

    grid_search.fit(X_data, y_data)
    results = pd.DataFrame(grid_search.cv_results_)

    now = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

    results.to_csv("gridsearch_results/rf_{}_search_{}.csv".format(method, now))


"""
Function to perform gridsearch for mlp model using inputted parameter options
trips: array of trip objects holding the data. trips will be called for all users if no object is used.
params: dictionary of parameter options to be used in gridsearch.
"""
def mlp_gridsearch(trips = None, params = None, method = 'grid', n_iter = 128):
    if trips is None:
        trips = get_trips_quick()

    data = get_tagged_dataset(trips, trim_end_zeros=True)

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
    
    if params is None:
        params = {
        'features': [rssi_features],
        'pca_features': [position_features],
        'scaler': [RobustScaler()],
        'pca': [PCA(n_components=2), PCA(n_components=4)],
        'hidden_layer_sizes': [(50,)],
        'learning_rate_init' : [0.001],
        'learning_rate' : ['constant'],
        'alpha' : [0.0001],
        'activation' : ['logistic'],
        'momentum' : [0.95],
        # 'batch_size': [16, 32, 64],
        'max_iter': [200, 500],
        'early_stopping': [True, False],
        'n_iter_no_change': [10, 20]
    }

    kf = StratifiedKFold(n_splits=5)

    X_data = [frame[all_features] for frame in data]
    y_data = [frame.iloc[0]["seat"] if frame.shape[0] > 0 else -1 for frame in data]

    if method == 'random':
        grid_search = RandomizedSearchCV(MLP(), params, n_iter = n_iter, cv = kf, scoring = mlp_prediction_scorer, n_jobs = -1, verbose = 5, return_train_score = True)
    elif method == 'halving':
        grid_search = HalvingGridSearchCV(MLP(), params, cv = kf, scoring = mlp_prediction_scorer, n_jobs = -1, verbose = 5, return_train_score = True)
    else:
        grid_search = GridSearchCV(MLP(), params, cv = kf, scoring = mlp_prediction_scorer, n_jobs = -1, verbose = 4, return_train_score = True)

    grid_search.fit(X_data, y_data)
    results = pd.DataFrame(grid_search.cv_results_)

    now = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

    results.to_csv("gridsearch_results/mlp_{}_search_{}.csv".format(method, now))

"""
Function to perform gridsearch for lstm model using inputted parameter options
trips: array of trip objects holding the data. trips will be called for all users if no object is used.
params: dictionary of parameter options to be used in gridsearch.
"""
def lstm_gridsearch(trips = None, params = None, method = 'grid', n_iter = 128):
    if trips is None:
        trips = get_trips_quick()
    
    data = get_tagged_dataset(trips, include_pretrip=False, trim_end_zeros=True)

    all_features = ['rssi_1', 'rssi_accuracy_1', 'rssi_2', 'rssi_accuracy_2', 'latitude',
       'longitude', 'speed', 'speedAcc', 'vertical_acc', 'altitude', 'course',
       'courseAcc', 'heading', 'horizontal_acc', 'attitude_pitch',
       'attitude_roll', 'attitude_yaw', 'rotation_rate_x', 'rotation_rate_y',
       'rotation_rate_z', 'gravity_accel_x', 'gravity_accel_y',
       'gravity_accel_z', 'user_accel_x', 'user_accel_y', 'user_accel_z']
    
    rssi_features = ['rssi_1', 'rssi_accuracy_1', 'rssi_2', 'rssi_accuracy_2']
    position_features = ['latitude',
       'longitude', 'speed', 'speedAcc', 'vertical_acc', 'altitude', 'course',
       'courseAcc', 'heading', 'horizontal_acc', 'attitude_pitch',
       'attitude_roll', 'attitude_yaw', 'rotation_rate_x', 'rotation_rate_y',
       'rotation_rate_z', 'gravity_accel_x', 'gravity_accel_y',
       'gravity_accel_z', 'user_accel_x', 'user_accel_y', 'user_accel_z']
    
    X_data = [frame[all_features] for frame in data]

    y_data = [frame.iloc[0]["seat"] if frame.shape[0] > 0 else -1 for frame in data]
    
    if params is None:
        params = {
            'features': [rssi_features],
            'scaler': [StandardScaler()],
            'pca': [PCA(n_components=2), PCA(n_components=3)],
            'pca_features': [position_features],
            'hidden_size': [50, 100],
            'lr' : [0.001, 0.01],
            'num_epochs': [10],
            'sub_sequence_length': [7],
            'batch_size': [10]
        }

    kf = StratifiedKFold(n_splits=5)
    
    if method == 'random':
        grid_search = RandomizedSearchCV(SklearnLSTMWrapper(), params, n_iter = n_iter, cv = kf, scoring = sequence_prediction_scorer, n_jobs = -1, verbose = 5, return_train_score = True)
    elif method == 'halving':
        grid_search = HalvingGridSearchCV(SklearnLSTMWrapper(), params, cv = kf, scoring = sequence_prediction_scorer, n_jobs = -1, verbose = 5, factor = 2, min_resources = 30, return_train_score = True)
    else:
        grid_search = GridSearchCV(SklearnLSTMWrapper(), params, cv = kf, scoring = sequence_prediction_scorer, n_jobs = -1, verbose = 3, return_train_score = True)

    grid_search.fit(X_data, y_data)
    results = pd.DataFrame(grid_search.cv_results_)

    now = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

    results.to_csv("gridsearch_results/mlp_{}_search_{}.csv".format(method, now))

    
def clean_gridsearch_results(df):
    new_df = df.copy()
    index_dict = {}

    for i, col in enumerate(new_df.columns):
        index_dict[col] = i
    
    new_order = []
    first_cols = ['rank_test_score', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score', 'mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
                  'split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score', 'split4_test_score', 'split0_train_score', 'split1_train_score', 'split2_train_score', 
                  'split3_train_score', 'split4_train_score']

    for col in first_cols:
        if col in index_dict.keys():
            new_order.append(index_dict[col])
            del index_dict[col]
    
    new_order.extend(index_dict.values())
    
    new_df = new_df.iloc[:, new_order]
    return new_df




"""
Function to get average performance metrics for each param option saved from gridsearch
Returns a dictionary conversion of a pandas dataframe

Args:
    file: a filepath string pointing to gridsearch results
"""
def get_gridsearch_splits(df):
    sum_cols = ['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time', 'split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score', 'split4_test_score', 'mean_test_score', 'std_test_score','rank_test_score',
                'split0_train_score', 'split1_train_score', 'split2_train_score', 'split3_train_score', 'split4_train_score', 'mean_train_score', 'std_train_score']
    ignore_cols = ['params', 'iter']
    summary_dict = {}

    for col in df.columns:
        if col not in (sum_cols + ignore_cols):
            if len(df[col].unique()) > 1:
                for val in df[col].unique():
                    filtered = df.loc[df[col] == val]
                    label = '{} = {}'.format(col, val)
                    temp_dict = {}
                    for c in sum_cols:
                        if c in filtered.columns:
                            temp_dict[c] = filtered[c].mean()
                    summary_dict[label] = temp_dict
    
    split_frame = pd.DataFrame(summary_dict)
    split_frame = split_frame.transpose()

    index_dict = {}
    for i, col in enumerate(split_frame.columns):
        index_dict[col] = i
    
    new_order = []
    first_cols = ['rank_test_score', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score', 'mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time']

    for col in first_cols:
        if col in index_dict.keys():
            new_order.append(index_dict[col])
            del index_dict[col]
    
    new_order.extend(index_dict.values())
    
    split_frame = split_frame.iloc[:, new_order]
    
    return split_frame