from TrajectoryAidedLearning.f110_gym.f110_env import F110Env
from TrajectoryAidedLearning.Utils.utils import *
from TrajectoryAidedLearning.Utils.HistoryStructs import VehicleStateHistory

from TrajectoryAidedLearning.Planners.PurePursuit import PurePursuit
from TrajectoryAidedLearning.Planners.AgentPlanners import AgentTester



import torch
import numpy as np
import time

# settings
SHOW_TRAIN = False
SHOW_TEST = False
# SHOW_TEST = True
VERBOSE = True
LOGGING = True


class TestSimulation():
    def __init__(self, run_file: str):
        self.run_data = setup_run_list(run_file)
        self.conf = load_conf("config_file")

        self.env = None
        self.num_agents = None
        self.target_planner = None
        self.adv_planners = None
        
        self.n_test_laps = None
        self.lap_times = None
        self.completed_laps = None
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

            if run.architecture == "PP": 
                planner = PurePursuit(self.conf, run)
            elif run.architecture == "fast": 
                planner = AgentTester(run, self.conf)
            else: raise AssertionError(f"Planner {run.planner} not found")

            if run.test_mode == "Std": self.target_planner = planner
            else: raise AssertionError(f"Test mode {run.test_mode} not found")

            self.vehicle_state_history = VehicleStateHistory(run, "Testing/")

            self.n_test_laps = run.n_test_laps
            self.lap_times = []
            self.completed_laps = 0

            eval_dict = self.run_testing()
            run_dict = vars(run)
            run_dict.update(eval_dict)

            save_conf_dict(run_dict)

            self.env.close_rendering()

    def run_testing(self):
        assert self.env != None, "No environment created"
        start_time = time.time()

        for i in range(self.n_test_laps):
            observation = self.reset_simulation()

            while not observation['colision_done'] and not observation['lap_done']:
                action = self.target_planner.plan(observation)
                observation = self.run_step(action)
                if SHOW_TEST: self.env.render('human_fast')

            self.planner.lap_complete()
            if observation['lap_done']:
                if VERBOSE: print(f"Lap {i} Complete in time: {observation['current_laptime']}")
                self.lap_times.append(observation['current_laptime'])
                self.completed_laps += 1

            if observation['colision_done']:
                if VERBOSE: print(f"Lap {i} Crashed in time: {observation['current_laptime']}")
                    

            if self.vehicle_state_history: self.vehicle_state_history.save_history(i, test_map=self.map_name)

        print(f"Tests are finished in: {time.time() - start_time}")

        success_rate = (self.completed_laps / (self.n_test_laps) * 100)
        if len(self.lap_times) > 0:
            avg_times, std_dev = np.mean(self.lap_times), np.std(self.lap_times)
        else:
            avg_times, std_dev = 0, 0

        print(f"Crashes: {self.n_test_laps - self.completed_laps} VS Completes {self.completed_laps} --> {success_rate:.2f} %")
        print(f"Lap times Avg: {avg_times} --> Std: {std_dev}")

        eval_dict = {}
        eval_dict['success_rate'] = float(success_rate)
        eval_dict['avg_times'] = float(avg_times)
        eval_dict['std_dev'] = float(std_dev)

        return eval_dict

    # this is an overide
    def run_step(self, actions):
        sim_steps = self.conf.sim_steps
        if self.vehicle_state_history: 
            self.vehicle_state_history.add_action(actions[0])
        self.prev_action = actions[0]

        sim_steps, done = sim_steps, False
        while sim_steps > 0 and not done:
            obs, step_reward, done, _ = self.env.step(actions)
            # print("In run_step, obs[collisions] :", obs["collisions"])
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
            observation['current_laptime'] = obs['lap_times'][0]
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

            observation['reward'] = 0.0
            ## Fixed collisions so shouldn't need this method anymore
            # if done and obs['lap_counts'][agent_id] == 0:
                # observation['colision_done'] = True
            if obs['collisions'][agent_id] == 1.:
                observation['colision_done'] = True

            if self.std_track is not None:
                if self.std_track.check_done(observation) and obs['lap_counts'][agent_id] == 0:
                    observation['colision_done'] = True

                if self.prev_obs is None: observation['progress'] = 0
                elif self.prev_obs['lap_done'] == True: observation['progress'] = 0
                else: observation['progress'] = max(self.std_track.calculate_progress_percent(state[0:2]), self.prev_obs['progress'])
                # self.racing_race_track.plot_vehicle(state[0:2], state[2])
                # taking the max progress
                

            if obs['lap_counts'][agent_id] == 1:
                observation['lap_done'] = True

            if self.reward and agent_id == 0: # ie. if target_planner
                observation['reward'] = self.reward(observation, self.prev_obs, self.prev_action)

            if self.vehicle_state_history:
                self.vehicle_state_history.add_state(obs['full_states'][agent_id])

            # Append agent_observation to total observations
            observations.append(observation)

        return observations

    def reset_simulation(self):
        reset_pose = np.zeros((self.num_agents, 3))
        # offset = np.concatenate((
        #                 np.arange(self.num_agents).reshape(1, -1),  # X axis
        #                 np.zeros((1, self.num_agents)),             # Y axis
        #                 np.zeros((1, self.num_agents))              # Z axis (unused I believe?)
        #                 ))
        x_offset = np.arange(self.num_agents) * 2.0
        y_offset = np.zeros(self.num_agents)
        y_offset[1::2] = np.ones(self.num_agents//2) * 0.6
        offset = np.concatenate((x_offset.reshape(1, -1), y_offset.reshape(1, -1), np.zeros((1, self.num_agents))), axis=0).T
        reset_pose -= offset

        obs, step_reward, done, _ = self.env.reset(reset_pose)

        if SHOW_TRAIN: self.env.render('human_fast')

        self.prev_obs = None
        observation = self.build_observation(obs, done)
        # self.prev_obs = observation
        if self.std_track is not None:
            self.std_track.max_distance = 0.0

        return observation


def main():
    # run_file = "PP_speeds"
    run_file = "PP_maps8"
    # run_file = "Eval_RewardsSlow"
    
    
    sim = TestSimulation(run_file)
    sim.run_testing_evaluation()


if __name__ == '__main__':
    main()


