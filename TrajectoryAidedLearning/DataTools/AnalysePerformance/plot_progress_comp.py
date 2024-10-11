import os
import sys
sys.path.insert(0, os.getcwd()) # hacky fix

from matplotlib import pyplot as plt
# plt.rc('font', family='serif')
# plt.rc('pdf',fonttype = 42)
# plt.rc('text', usetex=True)
plt.rcParams['pdf.use14corefonts'] = True

import numpy as np
import glob
import os
import argparse

import glob
from matplotlib.ticker import PercentFormatter
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, Arrow

from TrajectoryAidedLearning.DataTools.MapData import MapData
from TrajectoryAidedLearning.Utils.StdTrack import StdTrack 
from TrajectoryAidedLearning.Utils.RacingTrack import RacingTrack
from TrajectoryAidedLearning.Utils.utils import *
from matplotlib.ticker import MultipleLocator
from TrajectoryAidedLearning.DataTools.plotting_utils import *

# SAVE_PDF = False
SAVE_PDF = True

colors = ['red', 'blue', 'green', 'purple']
DT = 0.01 # defined by self.timestep in f110_env


def ensure_path_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


class AnalyseTestLapData:
    def __init__(self):
        self.path = None
        self.vehicle_name = None
        self.map_name = None
        self.states = None
        self.actions = None
        self.map_data = None
        self.std_track = None
        self.summary_path = None
        self.run_data = None
        self.num_agents = None
        self.n_test_laps = None
        self.lap_n = 0

    def explore_folder(self, run_file):
        self.run_data = setup_run_list(run_file)        
        self.sim_steps = load_conf("config_file").sim_steps

        self.num_agents = self.run_data[0].num_agents
        self.n_test_laps = self.run_data[0].n_test_laps
        path = "Data/Vehicles/" + run_file + "/"

        vehicle_folders = glob.glob(f"{path}*/")
        print(vehicle_folders)
        print(f"{len(vehicle_folders)} folders found")

        set = 1
        for j, folder in enumerate(vehicle_folders):
            print(f"Vehicle folder being opened: {folder}")
            self.process_folder(folder)

    def process_folder(self, folder):
        self.path = folder

        self.vehicle_name = self.path.split("/")[-2]
                
        self.map_name = self.vehicle_name.split("_")[4]
        if self.map_name == "f1":
            self.map_name += "_" + self.vehicle_name.split("_")[5]
        self.map_data = MapData(self.map_name)
        self.std_track = StdTrack(self.map_name)
        self.racing_track = RacingTrack(self.map_name)

        if not os.path.exists(self.path + "TestingProgressComp/"): 
            os.mkdir(self.path + "TestingProgressComp/")    
        for self.lap_n in range(self.n_test_laps):
            if not self.load_lap_data(): break # no more laps
            print(f"Test Lap {self.lap_n}:")
            print("     self.states.shape : ", self.states.shape)
            print("     self.actions.shape : ", self.actions.shape)
            self.plot_progress()


    def load_lap_data(self):
        try:
            data = np.array([np.load(self.path + f"Testing/agent_{agent_id}/Lap_{self.lap_n}_history_{self.vehicle_name}_{self.map_name}.npy") for agent_id in range(self.num_agents)])
        except Exception as e:
            print(e)
            print(f"No data for: " + f"Lap_{self.lap_n}_history_{self.vehicle_name}_{self.map_name}.npy")
            return 0
        self.states = data[:, :, :7]
        self.actions = data[:, :, 7:9]
        self.progresses = data[:, :, 9]

        return 1 # to say success

    
    def plot_progress(self): 
        save_path  = self.path + "TestingProgressComp/"
        
        plt.figure(1, figsize=(4.5, 2.3))
        agent_names = [f"{adv} (Adversary #{adv_idx + 1})" for adv_idx, adv in enumerate(self.run_data[0].adversaries)]
        agent_names.insert(0, f"{self.run_data[0].architecture} (Target)") 

        total_steps = self.progresses.shape[1]
        xs = np.linspace(0, total_steps * DT * self.sim_steps, total_steps)
        max_progress = np.max(self.progresses, axis=0)

        for agent_id in range(self.num_agents):
            progress = self.progresses[agent_id] - max_progress
            plt.plot(xs, progress, '-', color=colors[agent_id], linewidth=1, label=agent_names[agent_id], alpha=0.85)
        

        plt.xlabel("Time (s)")
        plt.ylabel("% Progress Behind Lead")
        plt.xlim(0.0, total_steps * DT * self.sim_steps)
        plt.legend(loc='center', bbox_to_anchor=(0.5, -0.52), ncol=3)
        plt.tight_layout()
        plt.grid()

        name = save_path + f"{self.vehicle_name}_velocity_map_{self.lap_n}_progress_comp"
        std_img_saving(name)

def set_limits(map_name):
    plt.xlim(20, 1500)
    plt.ylim(50, 520)
    

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


    
