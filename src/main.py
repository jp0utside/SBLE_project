from parse import *
from visualize import *
from matlab import *
import trip
from train_basic import *
import random
from random_forest import *

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

def test_random_forest(trips = None, split_seed=3, tree_seed=3):
    if not trips:
        trips = get_trips_quick()
    data = get_tagged_dataset(trips)
    data = pd.concat(data)
    features = ["rssi_1", "rssi_2", "rssi_accuracy_1", "rssi_accuracy_2", "speed_1", "speed_2", "speedAcc_1", "speedAcc_2"]
    X_train, X_test, y_train, y_test = train_test_split(data[features], data["seat"].tolist(), test_size = 0.2, random_state=split_seed)
    random_forest = RandomForest(random_state=tree_seed)
    random_forest.train(X_train, y_train)
    random_forest.test(X_test, y_test)
    random_forest.visualize_decision_bounds()

def main():
    pass
    trips = get_trips_quick()
    for i in range(3):
        for j in range(3):
            test_random_forest(trips, i, j)




if __name__ == "__main__":
    main()
