import os
import sys
sys.path.insert(0, os.getcwd()) # hacky fix

import matplotlib.pyplot as plt
import numpy as np
import glob
import argparse
import yaml

from TrajectoryAidedLearning.Utils.utils import *
from matplotlib.ticker import MultipleLocator
from TrajectoryAidedLearning.DataTools.plotting_utils import *

from TrajectoryAidedLearning.Utils.utils import *
from TrajectoryAidedLearning.DataTools.TrainingGraphs.TrainingUtils import *
from TrajectoryAidedLearning.DataTools.plotting_utils import *

SAVE_NAME = "training_progress_dist_111"

def plot_progress():
    names = [
        "SAC",
        "DreamerV3",
        "cRSSM",
        "cMask",
    ]
    runs = [
        "sac_multiagent_classic/SAC_111_15153030_Std_Cth_f1_esp_6_0_850_",
        "dreamerv3_multiagent_classic/DreamerV3_111_15153030_Std_Cth_f1_esp_6_0_850_",
        "cdreamer_multiagent_classic/cDreamer_111_15153030_Std_Cth_f1_esp_6_0_850_",
        "cbdreamer_multiagent_classic/cbDreamer_111_15153030_Std_Cth_f1_esp_6_0_850_",

    ]
    colors = [
        'gray',
        'red',
        'blue',
        'green',
    ]
    
    def find_progress(folder):
        rewards, lengths, progresses, _ = load_csv_data(folder)
        steps = np.cumsum(lengths[:-1]) / 1000
        # avg_progress = true_moving_average(progresses[:-1], 20)
        avg_progress = ewma(progresses[:-1])


        return avg_progress, steps    

    def explore_folder(run_name, run_idx=0):
        run_folders = glob.glob(f"Data/Vehicles/{run_name}*/")
        print("run_folder :", run_folders)

        progresses = []
        steps_list = []    
        for idx, folder in enumerate(run_folders):
            print(f"Vehicle folder being opened: {folder}")
            prog, steps = find_progress(folder)
            progresses.append(prog)
            steps_list.append(steps)
    
        xs = np.linspace(0, 100, 300)
        min, max, mean = convert_to_min_max_avg(steps_list, progresses, xs)
        # min, max, mean = convert_to_min_max_avg_iqm5(steps_list, progresses, xs)
        print("min, max, mean:", min.shape, max.shape, mean.shape)
        
        plt.cla()
        plt.clf()
        
        plt.plot(xs, mean, '-', color=colors[run_idx], linewidth=2, label=names[run_idx])
        plt.gca().fill_between(xs, min, max, color=colors[run_idx], alpha=0.2)

        plt.xlabel("Training Steps (x1000)")
        plt.ylabel("Track Progress %")
        plt.ylim(0, 1)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.grid()

        std_img_saving(f"Data/Vehicles/{run_name.split('/')[0]}/{SAVE_NAME}")

        return min, max, mean

    mins, maxs, means = [], [], []
    for idx, run in enumerate(runs):
        min, max, mean = explore_folder(run, run_idx=idx)
        mins.append(min)
        maxs.append(max)
        means.append(mean)
    
    plt.cla()
    plt.clf()
    xs = np.linspace(0, 100, 300)
    for run_idx in range(len(runs)):
        plt.plot(xs, means[run_idx], '-', color=colors[run_idx], linewidth=2, label=names[run_idx])
        plt.gca().fill_between(xs, mins[run_idx], maxs[run_idx], color=colors[run_idx], alpha=0.2)
    plt.xlabel("Training Steps (x1000)")
    plt.ylabel("Track Progress %")
    plt.ylim(0, 1)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.grid()
    std_img_saving(f"Data/Vehicles/{SAVE_NAME}")

plot_progress()

