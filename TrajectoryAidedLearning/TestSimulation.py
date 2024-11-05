import os
import sys
sys.path.insert(0, os.getcwd()) # hacky fix

from TrajectoryAidedLearning.f110_gym.f110_env import F110Env
from TrajectoryAidedLearning.Utils.utils import *

from TrajectoryAidedLearning.Planners.PurePursuit import PurePursuit
from TrajectoryAidedLearning.Planners.DisparityExtender import DispExt
from TrajectoryAidedLearning.Planners.TD3Planners import TD3Trainer, TD3Tester
from TrajectoryAidedLearning.Planners.SACPlanners import SACTrainer, SACTester
from TrajectoryAidedLearning.Planners.DreamerV2Planners import DreamerV2Trainer, DreamerV2Tester
from TrajectoryAidedLearning.Planners.DreamerV3Planners import DreamerV3Trainer, DreamerV3Tester

from TrajectoryAidedLearning.Utils.RewardSignals import *
from TrajectoryAidedLearning.Utils.StdTrack import StdTrack

from TrajectoryAidedLearning.Utils.HistoryStructs import VehicleStateHistory


import torch
import numpy as np
from math import ceil
import time

# settings
SHOW_TRAIN = False
SHOW_TEST = False
# SHOW_TEST = True
VERBOSE = True
LOGGING = True
GRID_X_COEFF = 2.0
GRID_Y_COEFF = 0.6

# TODO SOOOOOOO hacky fix
def select_reward_function(run, conf, std_track):
    reward = run.reward
    if reward == "Progress":
        reward_function = ProgressReward(std_track)
    elif reward == "Cth": 
        reward_function = CrossTrackHeadReward(std_track, conf)
    elif reward == "TAL":
        reward_function = TALearningReward(conf, run)
    else: raise Exception("Unknown reward function: " + reward)
        
    return reward_function

# TODO SOOOOOOO hacky fix
def select_agent(run, conf, architecture, train=True, init=False, ma_info=[0.0, 0.0]):
    agent_type = architecture if architecture is not None else "TD3"
    if agent_type == "PP":
        agent = PurePursuit(run, conf, init=init, ma_info=ma_info) 
    elif agent_type == "TD3":
        agent = TD3Trainer(run, conf, init=init) if train else TD3Tester(run, conf)
    elif agent_type == "SAC":
        agent = SACTrainer(run, conf, init=init) if train else SACTester(run, conf)
    elif agent_type == "DreamerV2":
        agent = DreamerV2Trainer(run, conf) if train else DreamerV2Tester(run, conf)
    elif agent_type == "DreamerV3":
        agent = DreamerV3Trainer(run, conf, init=init) if train else DreamerV3Tester(run, conf)
    elif agent_type == "DispExt":
        agent = DispExt(run, conf, ma_info=ma_info)
    else: raise Exception("Unknown agent type: " + agent_type)

    return agent


class TestSimulation():
    def __init__(self, run_file: str):
        self.run_data = setup_run_list(run_file)
        self.conf = load_conf("config_file")

        self.env = None
        self.num_agents = None
        self.target_position = None
        self.target_planner = None
        self.adv_planners = None
        
        self.n_test_laps = None
        self.lap_times = None
        self.completed_laps = None
        self.places = None
        self.progresses = None
        self.prev_obs = None
        self.prev_action = None

        self.std_track = None
        self.map_name = None
        self.reward = None
        self.noise_rng = None

        # flags 
        self.vehicle_state_history = None

    def run_testing_evaluation(self):
        for run in self.run_data:
            print(run)
            print("_________________________________________________________")
            print(run.run_name)
            print("_________________________________________________________")
            seed = run.random_seed + 10*run.n
            np.random.seed(seed) # repetition seed
            torch.use_deterministic_algorithms(True)
            torch.manual_seed(seed)

            if run.noise_std > 0:
                self.noise_std = run.noise_std
                self.noise_rng = np.random.default_rng(seed=seed)

            self.env = F110Env(
                map=run.map_name,
                num_agents=run.num_agents,
                )
            self.map_name = run.map_name
            self.num_agents = run.num_agents
            self.target_position = run.target_position

            self.std_track = StdTrack(run.map_name, num_agents=run.num_agents)
            self.reward = select_reward_function(run, self.conf, self.std_track)

            self.target_planner = select_agent(run, self.conf, run.architecture, train=False, init=False)
            
            self.n_test_laps = run.n_test_laps
            
            self.run_testing(run)

            conf = vars(self.conf)
            conf['path'] = run.path
            conf['run_name'] = run.run_name
            save_conf_dict(conf, "TrainingConfig")

            self.env.close_rendering()

            ###################

            # eval_dict = self.run_testing()
            # run_dict = vars(run)
            # run_dict.update(eval_dict)

            # save_conf_dict(run_dict)

            # self.env.close_rendering()

    def run_testing(self, run):
        if len(run.adversaries) == 0:
            ma_runlist = [[0.0, 0.0]]
        else:
            speed_val, la_val = run.ma_info[2:]
            speed_arange, la_arange = np.round(np.arange(-speed_val, speed_val + 1e-6, 0.1), 2), np.round(np.arange(-speed_val, speed_val + 1e-6, 0.1), 2)
            speed_grid, la_grid = np.meshgrid(speed_arange, la_arange, indexing='ij')
            ma_runlist = np.stack([speed_grid.ravel(), la_grid.ravel()], axis=1)

        for ma_idx, ma_info in enumerate(ma_runlist):
            self.adv_planners = [select_agent(run, self.conf, architecture, train=False, init=False, ma_info=ma_info) for architecture in run.adversaries]
            self.vehicle_state_history = [VehicleStateHistory(run, f"Testing/Testing_{ma_idx}/agent_{agent_id}") for agent_id in range(self.num_agents)]
            assert self.env != None, "No environment created"
            start_time = time.time()

            self.lap_times = []
            self.places = []
            self.progresses = []
            self.completed_laps = 0

            for i in range(self.n_test_laps):
                observations = self.reset_simulation()
                target_obs = observations[0]

                while not target_obs['colision_done'] and not target_obs['lap_done'] and not target_obs['current_laptime'] > self.conf.max_laptime:
                    self.prev_obs = observations
                    target_action = self.target_planner.plan(observations[0])
                    if len(self.adv_planners) > 0:
                        adv_actions = np.array([adv.plan(obs) if not obs['colision_done'] else [0.0, 0.0] for (adv, obs) in zip(self.adv_planners, observations[1:])])
                        actions = np.concatenate((target_action.reshape(1, -1), adv_actions), axis=0)
                    else:
                        actions = target_action.reshape(1, -1)
                    observations = self.run_step(actions)
                    target_obs = observations[0]

                    if SHOW_TEST: self.env.render('human_fast')

                self.target_planner.lap_complete()
                self.progresses.append(target_obs['progress'])
                if target_obs['lap_done']:
                    if VERBOSE: print(f"Lap {i} Complete in time: {target_obs['current_laptime']}")
                    self.lap_times.append(target_obs['current_laptime'])
                    self.places.append(target_obs['position'])

                    self.completed_laps += 1

                if target_obs['colision_done']:
                    if VERBOSE: print(f"Lap {i} Crashed in time: {target_obs['current_laptime']}")
                
                if target_obs['current_laptime'] > self.conf.max_laptime:
                    if VERBOSE: print(f"Lap {i} LapTimeExceeded in time: {target_obs['current_laptime']}")

                if self.vehicle_state_history: 
                    for vsh in self.vehicle_state_history:
                        vsh.save_history(i, test_map=self.map_name)
                        # vsh.save_history(f"test_{i}", test_map=self.map_name)


            print(f"Tests are finished in: {time.time() - start_time}")

            success_rate = (self.completed_laps / (self.n_test_laps) * 100)
            if len(self.lap_times) > 0:
                avg_times, times_std_dev = np.mean(self.lap_times), np.std(self.lap_times)
            else:
                avg_times, times_std_dev = 0, 0

            if len(self.places) > 0:
                avg_place, place_std_dev = np.mean(self.places), np.std(self.places)
            else:
                avg_place, place_std_dev = 0, 0

            if len(self.progresses) > 0:
                avg_progress, progress_std_dev = np.mean(self.progresses), np.std(self.progresses)
            else:
                avg_progress, progress_std_dev = 0, 0

            print(f"Crashes: {self.n_test_laps - self.completed_laps} VS Completes {self.completed_laps} --> {success_rate:.2f} %")
            print(f"Lap times Avg: {avg_times} --> Std: {times_std_dev}")
            print(f"Place Avg: {avg_place} --> Std: {place_std_dev}")
            print(f"Progress Avg: {avg_progress} --> Std: {progress_std_dev}")


            eval_dict = {}
            eval_dict['success_rate'] = float(success_rate)
            eval_dict['avg_times'] = float(avg_times)
            eval_dict['times_std_dev'] = float(times_std_dev)
            eval_dict['avg_place'] = float(avg_place)
            eval_dict['place_std_dev'] = float(place_std_dev)
            eval_dict['avg_progress'] = float(avg_progress)
            eval_dict['progress_std_dev'] = float(progress_std_dev)

            run_dict = vars(run)
            run_dict.update(eval_dict)

            save_conf_dict(run_dict, f"RunConfig_{ma_idx}")

            self.lap_times = []
            self.places = []
            self.progresses = []
            self.completed_laps = 0

    # this is an overide
    def run_step(self, actions):
        sim_steps = self.conf.sim_steps
        if self.vehicle_state_history: 
            for vsh, action in zip(self.vehicle_state_history, actions):
                vsh.add_action(action)
        self.prev_action = actions[0]

        sim_steps, done = sim_steps, False
        while sim_steps > 0 and not done:
            obs, step_reward, done, _ = self.env.step(actions)
            sim_steps -= 1
        
        observations = self.build_observation(obs, done)
        
        return observations

    def build_observation(self, obs, done):
        """Build observation

        Returns 
            state:
                [0]: x
                [1]: y
                [2]: yaw
                [3]: v
                [4]: steering
            scan:
                Lidar scan beams 
            
        """
        observations = []
        for agent_id in range(self.num_agents):
            observation = {}
            observation['current_laptime'] = obs['lap_times'][agent_id]
            observation['scan'] = obs['scans'][agent_id] #TODO: introduce slicing here
            
            if self.noise_rng:
                noise = self.noise_rng.normal(scale=self.noise_std, size=2)
            else: noise = np.zeros(2)
            pose_x = obs['poses_x'][agent_id] + noise[0]
            pose_y = obs['poses_y'][agent_id] + noise[1]
            theta = obs['poses_theta'][agent_id]
            linear_velocity = obs['linear_vels_x'][agent_id]
            steering_angle = obs['steering_deltas'][agent_id]
            state = np.array([pose_x, pose_y, theta, linear_velocity, steering_angle])

            observation['state'] = state
            observation['lap_done'] = False
            observation['colision_done'] = False
            observation['position'] = 0

            observation['reward'] = 0.0

            ## Fixed collisions so shouldn't need this method anymore
            # if done and obs['lap_counts'][agent_id] == 0:
                # observation['colision_done'] = True
            if obs['collisions'][agent_id] == 1.:
                observation['colision_done'] = True

            if self.std_track is not None:
                self.std_track.calculate_progress(agent_id, state[0:2])

                if (self.std_track.check_done(agent_id) and obs['lap_counts'][agent_id] == 0) \
                                    or (not self.prev_obs is None and self.prev_obs[agent_id]['colision_done']):
                    observation['colision_done'] = True


                if self.prev_obs is None: observation['progress'] = 0
                elif self.prev_obs[agent_id]['lap_done'] == True: observation['progress'] = 0
                else: observation['progress'] = max(self.std_track.calculate_progress_percent(agent_id), self.prev_obs[agent_id]['progress'])
                # self.racing_race_track.plot_vehicle(state[0:2], state[2])
                # taking the max progress
                

            if obs['lap_counts'][agent_id] == 1:
                observation['lap_done'] = True

            if self.reward and agent_id == 0: # ie. if target_planner
                reward_obs = None if self.prev_obs is None else self.prev_obs[agent_id]
                reward_action = None if self.prev_action is None else self.prev_action[agent_id]
                observation['reward'] = self.reward(observation, reward_obs, reward_action)

            if self.vehicle_state_history:
                self.vehicle_state_history[agent_id].add_state(obs['full_states'][agent_id])
                self.vehicle_state_history[agent_id].add_progress(observation['progress'])

            # Append agent_observation to total observations
            observations.append(observation)
        
        # Set the position values for each agent
        observations = self.score_positions(observations)

        for agent_id in range(self.num_agents):
            observations[agent_id]['overtaking'] = self.prev_obs[agent_id]['position'] - observations[agent_id]['position'] \
                                                        if not self.prev_obs is None else 0
            observations[agent_id]['reward'] += observations[agent_id]['overtaking']


        # if self.vehicle_state_history:
            # for agent_id in range(self.num_agents):
                # self.vehicle_state_history

        return observations

    def score_positions(self, observations):
        scores = self.std_track.s
        sorted_scores = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        for rank, agent_id in enumerate(sorted_scores):
            observations[agent_id]['position'] = rank + 1
        
        return observations

    def calc_offsets(self, num_adv):
        x_offset = np.arange(1, num_adv+1) * 5.5
        y_offset = np.zeros(num_adv)
        y_offset[::2] = np.ones(ceil(num_adv/2)) * 0.6
        offset = np.concatenate((x_offset.reshape(1, -1), y_offset.reshape(1, -1), np.zeros((1, num_adv))), axis=0).T

        return offset

    def reset_simulation(self):
        reset_pose = np.zeros((self.num_agents, 3))
        if self.num_agents > 1:
            reset_pose[:, 1] -= GRID_Y_COEFF/2

        num_adv = self.num_agents - 1
        adv_back = self.num_agents - self.target_position
        adv_front = num_adv - adv_back

        front_offset = self.calc_offsets(adv_front)
        back_offset = np.flip(self.calc_offsets(adv_back), axis=0)
        back_offset[:, 0] *= -1

        offset = np.concatenate((np.zeros((1, 3)), front_offset, back_offset), axis=0)        
        reset_pose += offset

        obs, step_reward, done, _ = self.env.reset(reset_pose)

        if SHOW_TRAIN: self.env.render('human_fast')

        self.prev_obs = None
        if self.std_track is not None:
            self.std_track.max_distance = np.zeros((self.num_agents)) - 999.9
            self.std_track.s = np.zeros((self.num_agents)) - 999.9

        observation = self.build_observation(obs, done)

        return observation


def main():
    # run_file = "dev"
    # run_file = "SAC_lr"
    # run_file = "SAC_gamma"
    # run_file = "SAC_singleagent"
    # run_file = "SAC_multiagent_stationary"
    # run_file = "SAC_multiagent_nonstationary"
    # run_file = "dreamerv3_lr"
    # run_file = "dreamerv3_singleagent"
    run_file = "dreamerv3_multiagent_stationary"
    # run_file = "dreamerv3_multiagent_nonstationary"
    
    sim = TestSimulation(run_file)
    sim.run_testing_evaluation()


if __name__ == '__main__':
    main()


