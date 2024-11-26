from parse import *
from visualize import *
from matlab import *
import trip
from train_basic import *
import random
from random_forest import *
from mlp import *
from rnn import *
from joint_model import *
from lstm import *
from eval import *
import time
from datetime import datetime
from sklearn.model_selection import GridSearchCV, StratifiedKFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer

bad_MM = [2, 8, 21]
cmap = {"front" : "blue", "middle" : "green", "back" : "red"}
hex_colors = [
    "#FF5733",  # Bright Red
    "#33FF57",  # Bright Green
    "#3357FF",  # Bright Blue
    "#FF33A8",  # Bright Pink
    "#33FFF6",  # Bright Cyan
    "#FFAF33",  # Bright Orange
    "#8D33FF",  # Bright Purple
    "#A8FF33",  # Bright Lime
    "#FF6F33",  # Bright Coral
    "#33FFAF",  # Bright Mint
    "#5733FF",  # Bright Violet
    "#FFD133",  # Bright Gold
    "#33D1FF",  # Bright Sky Blue
    "#FF33D1",  # Bright Magenta
    "#33FF33",  # Bright Green
    "#D133FF",  # Bright Lavender
    "#FFA833",  # Bright Amber
    "#33FFA8",  # Bright Aqua
    "#6F33FF",  # Bright Indigo
    "#33FFD1",  # Bright Turquoise
    "#FF3383",  # Bright Salmon
    "#33FF83",  # Bright Greenish
    "#3333FF",  # Bright Cobalt
    "#FF338F",  # Bright Pinkish
    "#33FFE1",  # Bright Light Cyan
    "#FF8F33",  # Bright Tangerine
    "#9F33FF",  # Bright Violet
    "#B8FF33",  # Bright Chartreuse
    "#FF7F33",  # Bright Tangerine
    "#33FFCF",  # Bright Teal
    "#6733FF",  # Bright Indigo
    "#FFDF33",  # Bright Mustard
    "#33B1FF",  # Bright Cerulean
    "#FF33F1",  # Bright Fuchsia
    "#33FF63",  # Bright Light Green
    "#FF3357"   # Bright Crimson
]

def gaussian_discovery(x=1):
    trips = get_trips_quick(user = "Zebra")
    for n in range(1,x+1):
        print(n)
        data = get_tagged_dataset(trips, n=n, include_pretrip = False, exclude_zeros = False, trim_end_zeros=True)
        data = pd.concat(data)
        front = data.loc[data["seat"] == "front"]
        middle = data.loc[data["seat"] == "middle"]
        back = data.loc[data["seat"] == "back"]
        fig, ax = plt.subplots()
        gmm = gaussian_mixture_model(front, features = ["rssi_1_adj", "rssi_2_adj"])
        graph_gaussian(front, gmm, fig, ax, "blue")
        gmm = gaussian_mixture_model(middle, features = ["rssi_1_adj", "rssi_2_adj"])
        graph_gaussian(middle, gmm, fig, ax, "green")
        gmm = gaussian_mixture_model(back, features = ["rssi_1_adj", "rssi_2_adj"])
        graph_gaussian(back, gmm, fig, ax, "red")
        plt.show()
        print()

def plot_various_data():
    trips = get_trips_quick(user = "Zebra")
    data = get_tagged_dataset(trips, normalize_acc = True, acc_val=10)
    data = pd.concat(data)
    for i, x in enumerate(data):
        graph_trip_rssi_diff(x)
        quick_scatter(x["rssi_1_adj"], x["rssi_2_adj"], xlabel="rssi_1_adj", ylabel="rssi_2_adj", title = "{} trip {}, seat: {}".format(trips[i].user, trips[i].trip_idx, trips[i].seat), color = cmap[trips[i].seat])
        quick_plot(x["timestamp"], x["rssi_diff_adj"], xlabel="timestamp", ylabel="rssi_diff_adj", title = "{} trip {}, seat: {}".format(trips[i].user, trips[i].trip_idx, trips[i].seat), color = cmap[trips[i].seat])
        quick_scatter(x["speed_1"], x["rssi_diff_adj"], xlabel="speed", ylabel="rssi_diff_adj", title = "{} trip {}, seat: {}".format(trips[i].user, trips[i].trip_idx, trips[i].seat), color = cmap[trips[i].seat])
        quick_plot(x["timestamp"], x["rssi_diff_adj"], xlabel="timestamp", ylabel="rssi_diff_adj", title = "{} trip {}, seat: {}".format(trips[i].user, trips[i].trip_idx, trips[i].seat), color = cmap[trips[i].seat])
        prog_mean = [sum(x.iloc[:idx+1]["rssi_diff_adj"])/(idx+1) for idx in range(x.shape[0])]
        quick_plot(x["timestamp"], prog_mean, xlabel="timestamp", ylabel="rssi_diff_adj", title = "{} trip {}, seat: {}".format(trips[i].user, trips[i].trip_idx, trips[i].seat), color = cmap[trips[i].seat])

def three_by_three_test(data = pd.DataFrame(), features = [], use_scaler = True, use_pca = True, n_components = 2):
    for tree_seed in range(3):
        for split_seed in range(3):
            print("Tree Seed: {}, Split Seed: {}".format(tree_seed, split_seed))
            test_random_forest(data, features, split_seed=split_seed, tree_seed=tree_seed, use_scaler=use_scaler, use_pca=use_pca, n_components=n_components)

#Procedure to compare accuracy between non-aggregated and aggregated feature sets
def featureset_comparison():
    all_features_raw = ['latitude_1', 'longitude_1', 'speed_1', 'speedAcc_1', 'vertical_acc_1', 'altitude_1', 'course_1', 'courseAcc_1', 'heading_1', 'horizontal_acc_1', 'rssi_1', 'rssi_accuracy_1', 'attitude_pitch_1', 'attitude_roll_1', 'attitude_yaw_1', 'rotation_rate_x_1', 'rotation_rate_y_1', 'rotation_rate_z_1', 'gravity_accel_x_1', 'gravity_accel_y_1', 'gravity_accel_z_1', 'user_accel_x_1', 'user_accel_y_1', 'user_accel_z_1', 'latitude_2', 'longitude_2', 'speed_2', 'speedAcc_2', 'vertical_acc_2', 'altitude_2', 'course_2', 'courseAcc_2', 'heading_2', 'horizontal_acc_2', 'rssi_2', 'rssi_accuracy_2', 'attitude_pitch_2', 'attitude_roll_2', 'attitude_yaw_2', 'rotation_rate_x_2', 'rotation_rate_y_2', 'rotation_rate_z_2', 'gravity_accel_x_2', 'gravity_accel_y_2', 'gravity_accel_z_2', 'user_accel_x_2', 'user_accel_y_2', 'user_accel_z_2', 'rssi_diff']
    all_features_agg = ['rssi_1', 'rssi_accuracy_1', 'rssi_2', 'rssi_accuracy_2', 'rssi_diff', 'latitude', 'longitude', 'speed', 'speedAcc', 'vertical_acc', 'altitude', 'course', 'courseAcc', 'heading', 'horizontal_acc', 'attitude_pitch', 'attitude_roll', 'attitude_yaw', 'rotation_rate_x', 'rotation_rate_y', 'rotation_rate_z', 'gravity_accel_x', 'gravity_accel_y', 'gravity_accel_z', 'user_accel_x', 'user_accel_y', 'user_accel_z']
    trips = get_trips_quick()
    # features = all_features_agg.copy()
    data = pd.concat(get_tagged_dataset(trips))

    print("Old feature set")
    features_old = all_features_raw.copy()
    features_old.remove("timestamp")
    features_old.remove("major_1")
    data = get_tagged_dataset(trips, aggregate_feats=False)
    data = pd.concat(data)
    test_random_forest(data, features_old, use_pca=False)
    print()

    print("New feature set")
    features_new = all_features_agg.copy()
    features_new.remove("timestamp")
    features_new.remove("major_1")
    data = get_tagged_dataset(trips)
    data = pd.concat(data)
    test_random_forest(data, features_new, use_pca=False)

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

def mlp_pca_gridsearch():
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
    
    X = [frame[rssi_features] for frame in data]
    y = [frame.iloc[0]["seat"] if frame.shape[0] > 0 else -1 for frame in data]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)

    model = SklearnLSTMWrapper(len(rssi_features), hidden_size=50)

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

    X = [frame[rssi_features] for frame in data]
    y = [frame["seat"] for frame in data]

    scores = model.KFold(X, y)

    return scores

def lstm_gridsearch(trips):
    if trips is None:
        trips = get_trips_quick()
    
    data = get_tagged_dataset(trips, include_pretrip=False, trim_end_zeros=True)
    all_features = ['rssi_1', 'rssi_accuracy_1', 'rssi_2', 'rssi_accuracy_2', 'weekminute', 'latitude',
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
    
    params = {
        'scaler': [RobustScaler(), StandardScaler()],
        'hidden_size': [50, 100, 200],
        'lr' : [0.001, 0.01],
        'num_epochs': [5, 10, 20, 50],
        'sub_sequence_length': [3, 5, 7, 10]
    }

    kf = StratifiedKFold(n_splits=5)
    
    grid_search = GridSearchCV(SklearnLSTMWrapper(), params, cv = kf, scoring = sequence_prediction_scorer, n_jobs = 1, verbose = 5)
    grid_search.fit(X_data, y_data)
    results = pd.DataFrame(grid_search.cv_results_)
    now = datetime.now()
    results.to_csv("gridsearch_results/lstm_gridsearch_results_{}.csv".format(now), index = False)


def main():
    pass
    all_features = ['rssi_1', 'rssi_accuracy_1', 'rssi_2', 'rssi_accuracy_2', 'weekminute', 'latitude',
       'longitude', 'speed', 'speedAcc', 'vertical_acc', 'altitude', 'course',
       'courseAcc', 'heading', 'horizontal_acc', 'attitude_pitch',
       'attitude_roll', 'attitude_yaw', 'rotation_rate_x', 'rotation_rate_y',
       'rotation_rate_z', 'gravity_accel_x', 'gravity_accel_y',
       'gravity_accel_z', 'user_accel_x', 'user_accel_y', 'user_accel_z']
    trips = get_trips_quick()
    lstm_gridsearch(trips)

    # trips = get_trips_quick(user = "Zebra")
    # data = get_tagged_dataset(trips)
    # data = pd.concat(data)
    # rssi_features = ['rssi_1', 'rssi_accuracy_1', 'rssi_2', 'rssi_accuracy_2']
    # X = data
    # y = data["seat"]
    # model = RandomForest(scaler = RobustScaler(), pca = None, features = rssi_features, pca_features = [])
    # raw_acc, maj_acc, first_correct, last_incorrect = analyze_rf(model, data)

    # trips = get_trips_quick(user = "Zebra")
    # data = get_tagged_dataset(trips)
    # data = pd.concat(data)
    # rssi_features = ['rssi_1', 'rssi_accuracy_1', 'rssi_2', 'rssi_accuracy_2']
    # X = data
    # y = data["seat"]
    # model = MLP(scaler = RobustScaler(), pca = None, features = rssi_features, pca_features= [])
    # preds, true, split, accs = get_mlp_preds(model, X, y)



    







    
    




if __name__ == "__main__":
    main()

