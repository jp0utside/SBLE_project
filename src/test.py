from parse import *
from filter import *
import trip
from random_forest import *
from mlp import *
from rnn import *
from joint_model import *
from lstm import *
from datetime import datetime
from sklearn.model_selection import GridSearchCV, StratifiedKFold, GroupKFold, StratifiedGroupKFold
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

def test_random_forest(data = [], features = [], split_seed=3, tree_seed=3, use_scaler = True, use_pca = True, n_components = 2, n_estimators = 100, criterion = "gini"):
    if len(data) == 0:
        trips = get_trips_quick()
        data = get_tagged_dataset(trips, trim_end_zeros=True)
    if not features:
        features = ["rssi_1", "rssi_2"]

    X_data = [frame[features] for frame in data]
    y_data = [frame["seat"] for frame in data]


    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2, random_state=split_seed)

    X_train = pd.concat(X_train)
    X_test = pd.concat(X_test)
    y_train = pd.concat(y_train)
    y_test = pd.concat(y_test)

    random_forest = RandomForest(random_state=tree_seed, n_estimators=n_estimators, use_pca=use_pca, use_scaler=use_scaler, n_components=n_components, criterion=criterion)
    # start = time.time()
    random_forest.train(X_train, y_train)
    # end = time.time()
    acc, cm = random_forest.test(X_test, y_test)
    print("Accuracy: ", acc)
    print(cm)
    # print("Time: ", (end - start))
    # importance_table = random_forest.get_importance_table(X_test, y_test)
    # print(importance_table)
    # print(importance_table.to_dict())
    # print()
    return acc

def test_random_forest_new(trips = None, model = None):
    if trips is None:
        trips = get_trips_quick()
    
    if model is None:
        model = RandomForest(scaler = StandardScaler(), features=rssi_features, pca = None)
    
    data = get_tagged_dataset(trips)
    data = pd.concat(data)
    
    X_data = data[model.features]
    y_data = data["seat"]
    groups = data["group"]

    kf = StratifiedGroupKFold(n_splits=2)

    split_gen = kf.split(X_data, y_data, groups)

    train_idx, test_idx = next(split_gen)

    X_train = X_data.iloc[train_idx]
    y_train = y_data.iloc[train_idx]
    X_test = X_data.iloc[test_idx]
    y_test = y_data.iloc[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc, cm = accuracy_score(y_test, y_pred)

    print("acc: ", acc)
    print(cm)
    
    return acc, cm



def kfold_random_forest(data = [], features = [], split_seed=3, tree_seed=3, use_scaler = True, use_pca = True, n_components = 2, n_estimators = 100, criterion = "gini", class_weight = None):
    if len(data) == 0:
        trips = get_trips_quick()
        data = get_tagged_dataset(trips, trim_end_zeros=True)
    if not features:
        features = ["rssi_1", "rssi_2"]
    
    print("Features: ", features)

    X_data = [frame[features] for frame in data]
    y_data = [frame["seat"] for frame in data]

    random_forest = RandomForest(random_state=tree_seed, n_estimators=n_estimators, use_pca=use_pca, use_scaler=use_scaler, n_components=n_components, criterion=criterion, class_weight=class_weight)

    scores = random_forest.kfold(X_data, y_data)

    print(scores)
    print("Average: {}".format(sum(scores)/len(scores)))

    return scores

def kfold_random_forest_new(trips = None, model = None):
    if trips is None:
        trips = get_trips_quick()
    
    if model is None:
        model = RandomForest(scaler = StandardScaler(), features=rssi_features, pca = None)
    
    data = get_tagged_dataset(trips, trim_end_zeros = True)
    data = pd.concat(data)

    X_data = data[model.features]
    y_data = data["seat"]
    groups = data["group"]

    kf = StratifiedGroupKFold(n_splits=5)

    scores = []
    confs = []

    for i, (train_index, test_index) in enumerate(kf.split(X_data, y_data, groups)):
        split_model = clone(model)
        X_train = X_data.iloc[train_index].copy()
        X_test = X_data.iloc[test_index].copy()
        y_train = y_data.iloc[train_index].copy()
        y_test = y_data.iloc[test_index].copy()

        split_model.fit(X_train, y_train)
        y_pred = split_model.predict(X_test)

        acc, cm = accuracy_score(y_test, y_pred)
        scores.append(acc)
        confs.append(cm)
    
    print("Scores: ", scores)
    print("Avg: ", sum(scores)/len(scores))
    print("CMs: ", confs)

    return scores, confs

def kfold_mlp_new(trips = None, model = None):
    if trips is None:
        trips = get_trips_quick()
    
    if model is None:
        model = MLP(scaler = StandardScaler(), features=rssi_features, pca = None)
    
    data = get_tagged_dataset(trips, trim_end_zeros = True)
    data = pd.concat(data)

    X_data = data[model.features]
    y_data = data["seat"]
    groups = data["group"]

    kf = StratifiedGroupKFold(n_splits=5)

    scores = []

    for i, (train_index, test_index) in enumerate(kf.split(X_data, y_data, groups)):
        split_model = clone(model)
        X_train = X_data.iloc[train_index].copy()
        X_test = X_data.iloc[test_index].copy()
        y_train = y_data.iloc[train_index].copy()
        y_test = y_data.iloc[test_index].copy()

        split_model.fit(X_train, y_train)
        y_pred = split_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        scores.append(acc)
    
    print("Scores: ", scores)
    print("Avg: ", sum(scores)/len(scores))
    return(scores)



def random_forest_gridsearch(data, pipeline_options, param_options, features):
    groups = np.repeat(range(len(data)), [len(df) for df in data])

    all_data = pd.concat(data)
    X_data = all_data[features]
    y_data = all_data["seat"]

    kf = GroupKFold(n_splits=5)

    for pipeline, params in zip(pipeline_options, param_options):
        grid_search = GridSearchCV(pipeline, params, cv = kf, scoring="accuracy", n_jobs = -1, verbose = 5)
        grid_search.fit(X_data, y_data, groups=groups)

        results = pd.DataFrame(grid_search.cv_results_)
        now = datetime.now()
        results.to_csv("gridsearch_results/random_forest_gridsearch_results_{}.csv".format(now), index = False)

def mlp_gridsearch(data, pipelines, param_options, features):
    groups = np.repeat(range(len(data)), [len(df) for df in data])

    all_data = pd.concat(data)
    X_data = all_data[features]
    y_data = all_data["seat"]

    kf = GroupKFold(n_splits=5)

    for pipeline, params in zip(pipelines, param_options):
        grid_search = GridSearchCV(pipeline, params, cv = kf, scoring = "accuracy", n_jobs = -1, verbose = 5)
        grid_search.fit(X_data, y_data, groups=groups)

        results = pd.DataFrame(grid_search.cv_results_)
        now = datetime.now()
        results.to_csv("gridsearch_results/mlp_gridsearch_results_{}.csv".format(now), index = False)

def pca_test(trips = None):
    if trips is None:
        trips = get_trips_quick()
    
    data = get_tagged_dataset(trips, include_pretrip = False, trim_end_zeros = True)
    data = pd.concat(data)
    
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
    
    clf = MLP(scaler = RobustScaler(), pca = PCA(n_components = 4), features = rssi_features, pca_features = pca_features)

    seats = data["seat"].copy()
    groups = data["group"].copy()

    scores = clf.GroupKFold(data, seats, groups)

    return scores

def mlp_pca_gridsearch(trips = None):
    if trips is None:
        trips = get_trips_quick()

    data = get_tagged_dataset(trips, trim_end_zeros=True)
    data = pd.concat(data)

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

    kf = StratifiedGroupKFold(n_splits=5)
    # kf = GroupKFold(n_splits = 5)

    X_data = data[all_features]
    y_data = data["seat"]
    groups = data["group"]


    params = {
        'features': [rssi_features],
        'pca_features': [(rssi_features + position_features), position_features],
        'scaler': [RobustScaler()],
        'pca': [PCA(n_components=2), PCA(n_components=3), PCA(n_components=4)],
        'hidden_layer_sizes': [(50,), (100,), (50,50), (100, 50)],
        'learning_rate_init' : [0.001],
        'learning_rate' : ['constant'],
        'alpha' : [0.0001],
        'activation' : ['relu', 'tanh', 'logistic'],
        'momentum' : [0.8, 0.9, 0.95],
        # 'batch_size': [16, 32, 64],
        'max_iter': [200, 500, 1000],
        'early_stopping': [True, False],
        'n_iter_no_change': [10, 20, 50]
    }

    grid_search = GridSearchCV(MLP(), params, cv = kf, scoring = "accuracy", n_jobs = -1, verbose = 5)
    grid_search.fit(X_data, y_data, groups=groups)
    results = pd.DataFrame(grid_search.cv_results_)
    now = datetime.now()
    results.to_csv("gridsearch_results/mlp_pca_gridsearch_results_{}.csv".format(now), index = False)

def test_lstm(trips = None):
    if trips is None:
        trips = get_trips_quick()

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
    
    X = [frame[all_features] for frame in data]
    y = [frame.iloc[0]["seat"] if frame.shape[0] > 0 else -1 for frame in data]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)

    model = SklearnLSTMWrapper(all_features, scaler = StandardScaler(), pca_features=all_features, pca = PCA(n_components=3), hidden_size=50, batch_size=10)

    model.fit(X_train, y_train)
    acc = sequence_prediction_scorer(model, X_test, y_test)
    # preds, seq_idxs = model.predict(X_test)

    # y_true = [y_test[idx].iloc[0] for idx in seq_idxs]

    # acc = accuracy_score(y_true, preds)
    print("acc: ", acc)
    # return preds, y_test, seq_idxs

def lstm_kfold(trips = None):
    if trips is None:
        trips = get_trips_quick()

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
    
    model = SklearnLSTMWrapper(len(rssi_features), hidden_size=50)

    X = [frame[rssi_features] for frame in data]
    y = [frame["seat"] for frame in data]

    scores = model.KFold(X, y)

    return scores

def lstm_gridsearch(trips = None, params = None):
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
            'pca': [None, PCA(n_components=2), PCA(n_components=3)],
            'pca_features': [all_features, position_features],
            'hidden_size': [50, 100],
            'lr' : [0.001, 0.01],
            'num_epochs': [10, 25],
            'sub_sequence_length': [7, 10],
            'batch_size': [10, 25]
        }

    kf = StratifiedKFold(n_splits=5)
    
    grid_search = GridSearchCV(SklearnLSTMWrapper(), params, cv = kf, scoring = sequence_prediction_scorer, n_jobs = -1, verbose = 5)
    grid_search.fit(X_data, y_data)
    results = pd.DataFrame(grid_search.cv_results_)
    now = datetime.now()
    results.to_csv("gridsearch_results/lstm_gridsearch_results_{}.csv".format(now), index = False)

def run_gridsearch():
    trips = get_trips_quick()
    data = get_tagged_dataset(trips, trim_end_zeros=True)

    mlp_pipelines = [
        # Pipeline([
        #     ('scaler', StandardScaler()),
        #     ('mlp', MLPClassifier())
        # ]),
        Pipeline([
            ('scaler', RobustScaler()),
            ('mlp', MLPClassifier())
        ])
    ]

    mlp_pipeline_options = [
    # {
    #     'scaler__with_mean': [True],
    #     'scaler__with_std': [True] 
    # },
    {
        'scaler__with_centering': [True],
        'scaler__with_scaling': [True]
    }
    ]

    mlp_param_options = {
        'mlp__hidden_layer_sizes': [(50,), (100,), (50,50), (100, 50)],
        'mlp__learning_rate_init' : [0.001],
        'mlp__learning_rate' : ['constant'],
        'mlp__alpha' : [0.0001],
        'mlp__activation' : ['relu', 'tanh', 'logistic'],
        'mlp__momentum' : [0.8, 0.9, 0.95],
        # 'mlp__batch_size': [16, 32, 64],
        'mlp__max_iter': [200, 500, 1000],
        'mlp__early_stopping': [True, False],
        'mlp__n_iter_no_change': [10, 20, 50]
    }

    mlp_params = [{**mlp_param_options, **option} for option in mlp_pipeline_options]

    # print(mlp_params)

    features = ["rssi_1", "rssi_accuracy_1", "rssi_2", "rssi_accuracy_2"]
    mlp_gridsearch(data, mlp_pipelines, mlp_params, features)

def test_rnn_on_bus(trips = None):
    if trips is None:
        trips = get_trips_quick()

    features = ['rssi_1', 'rssi_accuracy_1', 'rssi_2', 'rssi_accuracy_2', 'weekminute', 'latitude',
       'longitude', 'speed', 'speedAcc', 'vertical_acc', 'altitude', 'course',
       'courseAcc', 'heading', 'horizontal_acc', 'attitude_pitch',
       'attitude_roll', 'attitude_yaw', 'rotation_rate_x', 'rotation_rate_y',
       'rotation_rate_z', 'gravity_accel_x', 'gravity_accel_y',
       'gravity_accel_z', 'user_accel_x', 'user_accel_y', 'user_accel_z',
       'magnetic_field_x', 'magnetic_field_y', 'magnetic_field_z']
    
    X_data, y_data = get_on_bus_data(trips, features)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2, random_state=3)

    clf = RNN(len(features))

    eval_data = clf.train_test(X_train, y_train, X_test, y_test)

    return eval_data, X_test

def test_rnn_seat_loc(trips = None):
    if trips is None:
        trips = get_trips_quick()

    features = ['rssi_1', 'rssi_accuracy_1', 'rssi_2', 'rssi_accuracy_2', 'weekminute', 'latitude',
       'longitude', 'speed', 'speedAcc', 'vertical_acc', 'altitude', 'course',
       'courseAcc', 'heading', 'horizontal_acc', 'attitude_pitch',
       'attitude_roll', 'attitude_yaw', 'rotation_rate_x', 'rotation_rate_y',
       'rotation_rate_z', 'gravity_accel_x', 'gravity_accel_y',
       'gravity_accel_z', 'user_accel_x', 'user_accel_y', 'user_accel_z',
       'magnetic_field_x', 'magnetic_field_y', 'magnetic_field_z']
    
    X_data, y_data = get_seat_loc_data_padded(trips, features)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2, random_state=3)

    clf = RNN(len(features), output_size=3)

    eval_data = clf.train_test(X_train, y_train, X_test, y_test)

    return eval_data, X_test

def test_joint_model(trips = None):
    if trips is None:
        trips = get_trips_quick()

    rssi_features = ['rssi_1', 'rssi_accuracy_1', 'rssi_2', 'rssi_accuracy_2']
    conf_features = ['rssi_1', 'rssi_accuracy_1', 'rssi_2', 'rssi_accuracy_2', 'weekminute', 'latitude',
       'longitude', 'speed', 'speedAcc', 'vertical_acc', 'altitude', 'course',
       'courseAcc', 'heading', 'horizontal_acc', 'attitude_pitch',
       'attitude_roll', 'attitude_yaw', 'rotation_rate_x', 'rotation_rate_y',
       'rotation_rate_z', 'gravity_accel_x', 'gravity_accel_y',
       'gravity_accel_z', 'user_accel_x', 'user_accel_y', 'user_accel_z',
       'magnetic_field_x', 'magnetic_field_y', 'magnetic_field_z']
    
    data = get_tagged_dataset(trips, include_pretrip=True, trim_end_zeros=False)
    data = pad_data(data)


    train_data, temp_data = train_test_split(data, test_size = 0.2, random_state = 3)
    val_data, test_data = train_test_split(temp_data, test_size = 0.5, random_state = 3)

    conf_train = np.stack([frame[conf_features] for frame in train_data])
    rssi_train = np.stack([frame[rssi_features] for frame in train_data])
    bus_train = np.stack([frame["on_bus"] for frame in train_data])
    seat_train = np.stack([frame["seat"] for frame in train_data])

    conf_val = np.stack([frame[conf_features] for frame in val_data])
    rssi_val = np.stack([frame[rssi_features] for frame in val_data])
    bus_val = np.stack([frame["on_bus"] for frame in val_data])
    seat_val = np.stack([frame["seat"] for frame in val_data])

    conf_test = np.stack([frame[conf_features] for frame in test_data])
    rssi_test = np.stack([frame[rssi_features] for frame in test_data])
    bus_test = np.stack([frame["on_bus"] for frame in test_data])
    seat_test = np.stack([frame["seat"] for frame in test_data])

    train_set = JointDataset(conf_train, rssi_train, bus_train, seat_train)
    val_set = JointDataset(conf_val, rssi_val, bus_val, seat_val)
    test_set = JointDataset(conf_test, rssi_test, bus_test, seat_test)

    train_loader = DataLoader(train_set, batch_size = 2, shuffle = True)
    val_loader = DataLoader(val_set, batch_size = 2, shuffle = False)
    test_loader = DataLoader(test_set, batch_size = 1, shuffle = False)

    model = JointModel(len(conf_features), len(rssi_features))

    model = model.train_model(train_loader, val_loader)

    test_results = model.test_model(test_loader)

    return test_results

def get_gridsearch_splits(file):
    df = pd.read_csv(file)
    sum_cols = ['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time', 'split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score', 'split4_test_score', 'mean_test_score', 'std_test_score','rank_test_score']
    ignore_cols = ['params']
    summary_dict = {}

    for col in df.columns:
        if col not in (sum_cols + ignore_cols):
            for val in df[col].unique():
                filtered = df.loc[df[col] == val]
                label = '{} = {}'.format(col, val)
                temp_dict = {}
                for c in sum_cols:
                    temp_dict[c] = filtered[c].mean()
                summary_dict[label] = temp_dict
    
    split_frame = pd.DataFrame(summary_dict)
    split_frame = split_frame.transpose()
    
    return summary_dict