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


class AnalyseTestLapData:
    def __init__(self):
        self.run_data = None
        self.conf = None

    def explore_folder(self, run_file):
        self.run_data = setup_run_list(run_file)
        self.conf = load_conf("config_file")
        self.n = self.run_data[-1].n + 1

        self.num_agents = self.run_data[0].num_agents
        self.n_test_laps = self.run_data[0].n_test_laps
        path = "Data/Vehicles/" + run_file + "/"

        vehicle_folders = glob.glob(f"{path}*/")
        run_names = [folder[:-2] for folder in vehicle_folders[:int(len(vehicle_folders)/self.n)]]
        run_folders = [glob.glob(f"{run_name}*/") for run_name in run_names]
        run_folders = [run_folders[0]]
        print(run_folders)
        print(f"{len(vehicle_folders)} folders found")

        for run_num, run_folder in enumerate(run_folders):
            run_succ = np.empty((0, 7, 7))
            for idx, folder in enumerate(run_folder):
                print(f"Vehicle folder being opened: {folder}")
                indv_succ = self.find_succ(folder)
                run_succ = np.concatenate([run_succ, indv_succ])
                self.plot_succ_mat(indv_succ, save_path=f'{folder}progress_{idx}')
            self.plot_succ_mat(np.mean(run_succ, axis=0).reshape(1, 7, 7), save_path=f'{run_names[run_num]}progress')



    def plot_succ_mat(self, data, save_path=None):
        # Reshape data
        data = data.squeeze(0)

        # Plot using matplotlib
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(data, cmap='magma_r', vmin=0.0, vmax=1.0)

        # Add color bar to show the scale
        fig.colorbar(cax)

        # Annotate each cell with the numeric value
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f'{data[i, j]:.2f}', va='center', ha='center', color='black' if data[i, j] < 0.5 else 'white')

        # Draw a box around the middle 3x3 area
        rect = plt.Rectangle((2 - 0.5, 2 - 0.5), 3, 3, edgecolor='green', facecolor='none', linewidth=2, linestyle='--')
        ax.add_patch(rect)

        # Set labels and title
        ax.grid(color='white', linewidth=0)
        ax.set_xticklabels(np.arange(-4, 4) / 10)
        ax.set_yticklabels(np.arange(-4, 4) / 10)
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xlabel('Speed Coeff')
        ax.set_ylabel('Steering Coeff')
        # ax.set_title('Success Rate')

        std_img_saving(save_path)

    def find_succ(self, folder):
        indv_succ = np.zeros((7, 7))
        for i in range(0, 49):
            yaml_path = f"{folder}RunConfig_{i}_record.yaml"
            with open(yaml_path) as file:
                run_config = yaml.safe_load(file)
                indv_succ[i // 7, i % 7] = run_config['avg_progress']

        return indv_succ.reshape((1, 7, 7))



def analyse_folder(run_file):
    TestData = AnalyseTestLapData()
    TestData.explore_folder(run_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run-file', default='dev', type=str)
    args = parser.parse_args()

    analyse_folder(args.run_file)

if __name__ == "__main__":
    main()