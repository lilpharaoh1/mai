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

        path = "Data/Vehicles/" + run_file + "/"

        vehicle_folders = sorted(glob.glob(f"{path}*/"))
        run_names = [folder[:-2] for folder in vehicle_folders[::self.n]]
        run_folders = [sorted(glob.glob(f"{run_name}*/")) for run_name in run_names]
        print(f"{len(vehicle_folders)} folders found")

        in_dist = np.full((1, 7, 7), False, dtype=bool)
        in_dist[:, 2:5, 2:5] = True
        for run_num, run_folder in enumerate(run_folders):
            self.num_agents = self.run_data[len(run_folders) - 1 - run_num].num_agents
            self.n_test_laps = self.run_data[len(run_folders) - 1 - run_num].n_test_laps
            run_succ = np.empty((0, 7, 7))
            for idx, folder in enumerate(run_folder):
                try:
                    print(f"Vehicle folder being opened: {folder}")
                    indv_succ = self.find_succ(folder)
                    run_succ = np.concatenate([run_succ, indv_succ])
                    self.plot_succ_mat(indv_succ, save_path=f'{folder}times_{idx}')
                except:
                    print(f"NOTE -> not counting folder {folder}")
            run_succ[run_succ == 0.0] = np.nan
            self.plot_succ_mat(np.nanmean(run_succ, axis=0).reshape(1, 7, 7), save_path=f'{run_names[run_num]}times')
            iid = np.nanmean(run_succ, axis=0).reshape(1, 7, 7)[in_dist]
            ood = np.nanmean(run_succ, axis=0).reshape(1, 7, 7)[~in_dist]

            print(f"In-distribution: {np.nanmean(iid)} +- {np.nanstd(iid)}")
            print(f"Out-of-distribution: {np.nanmean(ood)} +- {np.nanstd(ood)}")


    def plot_succ_mat(self, data, save_path=None):
        # Reshape data
        data = data.squeeze(0)

        # Plot using matplotlib
        fig, ax = plt.subplots(figsize=(8, 6))
        masked_data = np.copy(data)
        masked_data[masked_data == 0] = self.num_agents
        masked_data[masked_data == np.nan] = self.num_agents
        masked_data[np.isnan(masked_data)] = self.num_agents

        # Add color bar to show the scale
        cax = ax.matshow(masked_data, cmap='magma', vmin=1.0, vmax=self.num_agents)
        fig.colorbar(cax)

        # Annotate each cell with the numeric value
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                time = data[i, j]
                if time == 0.0 or time == np.nan or np.isnan(time):
                    ax.text(j, i, '-.--', va='center', ha='center', color='black')
                else:
                    ax.text(j, i, f'{time:.2f}', va='center', ha='center', color='black' if time > ((self.num_agents) / 2) else 'white')                    
                    

        # Draw a box around the middle 3x3 area
        rect = plt.Rectangle((2 - 0.5, 2 - 0.5), 3, 3, edgecolor='green', facecolor='none', linewidth=2, linestyle='--')
        ax.add_patch(rect)

        # Set labels and title
        ax.grid(color='white', linewidth=0)
        ax.set_xticklabels(np.arange(-4, 4) / 10)
        ax.set_yticklabels(np.arange(-4, 4) / 10)
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xlabel('Steering Coeff')
        ax.set_ylabel('Speed Coeff')
        # ax.set_title('Success Rate')

        std_img_saving(save_path)

    def find_succ(self, folder):
        indv_succ = np.zeros((7, 7))
        for i in range(0, 49):
            yaml_path = f"{folder}RunConfig_{i}_record.yaml"
            with open(yaml_path) as file:
                run_config = yaml.safe_load(file)
                indv_succ[i // 7, i % 7] = run_config['avg_times']

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