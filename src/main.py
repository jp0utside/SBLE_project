from parse import *
from visualize import *
from matlab import *
import trip
from train_basic import *
import random
from random_forest import *
from mlp import *
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
from test import *

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

def main():
    pass
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
    
    params = {
            'features': [rssi_features],
            'scaler': [StandardScaler()],
            'pca': [PCA(n_components = 4)],
            'pca_features': [position_features],
            'hidden_size': [25, 50],
            'lr' : [0.001, 0.01],
            'num_epochs': [5, 10],
            'sub_sequence_length': [5, 10, 15],
            'batch_size': [10, 25],
            'num_layers': [1,2],
            'optimizer': ['adam'], #'sgd'
            'bidirectional': [True, False],
            'dropout': [0.2, 0.3, 0.35]
            # 'momentum': [0.8, 0.9, 0.95]
    }

    trips = get_trips_quick()
    lstm_gridsearch(trips, params, method = 'halving')


"""
-------------------------------------------------------------------------------------
ARCHIVE
-------------------------------------------------------------------------------------
"""

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


if __name__ == "__main__":
    main()

