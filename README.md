# Model-Based Reinforcement Learning for Multi-Agent Autonomous Racing
## Emran Yasser Moustafa - 20332041

This repo contains the code I used for my MAI thesis project "Model-Based Reinforcement Learning for Multi-Agent Autonomous Racing".

As a starting point, I forked the F1Tenth gym environment used in the paper "[High-speed Autonomous Racing using Trajectory-aided Deep Reinforcement Learning](https://ieeexplore.ieee.org/document/10182327)". My implementation built upon this work by including a configurable framework for multi-agent racingg and by implementing the SAC, DreamerV3, cRSSM and cMask algorithms. I also implemented two context-parameterised adversarial agents; the Classic agent and the Disparity Extender agent. For details on these adversaries, the RL algorithms we used and the F1Tenth environment, please refer to my thesis document.  

![](Data/tal_calculation.png)

Training agents with our reward signal results in significatly improved training performance.
The most noteable performance difference is at high-speeds where previous rewards failed.

![](Data/TAL_vs_baseline_reward.png)

The improved training results in higher average progrresses at high speeds.

![](Data/tal_progress.png)

# Result Generation

The results in the paper are generated through a two step process of:
1. Train and test the agents
2. Process and plot the data

For every test:
- Run calculate_statistics
- Run calculate_averages

## Tests:

### Maximum Speed Investigation

- Aim: Understand how performance changes with different speeds.
- Config files: CthSpeeds, TAL_speeds 
- Results: 
    - Training graph: Cth_TAL_speeds_TrainingGraph
    - Lap times and % success: Cth_TAL_speeds_Barplot

### 6 m/s Performance Comparision 

- Aim: Compare the baseline and TAL on different maps with a maximum speed of 6 m/s.
- Config file: Cth_maps, TAL_maps
- Results:
    - Training graphs: TAL_Cth_maps_TrainingGraph
    - Lap times and success bar plot: TAL_Cth_maps_Barplot

### Speed Profile Analysis 

- Aim: Study the speed profiles
- Requires the pure pursuit (PP_speeds) results
- Results:
    - Trajectories: GenerateVelocityProfiles, set the folder to TAL_speeds
    - Speed profile pp TAL: TAL_speed_profiles
    - Speed profile x3: TAL_speed_profiles 
    - Slip profile: TAL_speed_profiles

### Comparison with Literatures

- Aim: Compare our method with the literature
- Results:
    - Bar plot: LiteratureComparison
- Note that the results from the literature are hard coded.

![](Data/animation.gif)


## Citation

If you find this work useful, please consider citing:
```
@ARTICLE{10182327,
    author={Evans, Benjamin David and Engelbrecht, Herman Arnold and Jordaan, Hendrik Willem},
    journal={IEEE Robotics and Automation Letters}, 
    title={High-Speed Autonomous Racing Using Trajectory-Aided Deep Reinforcement Learning}, 
    year={2023},
    volume={8},
    number={9},
    pages={5353-5359},
    doi={10.1109/LRA.2023.3295252}
}
```
