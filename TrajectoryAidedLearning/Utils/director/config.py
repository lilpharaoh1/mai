from dataclasses import dataclass, field
# from hydra.core.config_store import ConfigStore
from typing import Optional, Any, Tuple, Dict
from enum import Enum
import numpy as np
import torch.nn as nn


class Device(str, Enum):
    CUDA = "cuda"
    CPU = "cpu"


@dataclass
class MLPConfig:
    n_layers: int
    layer_size: int
    input_size: Optional[int] = None
    output_size: Optional[int] = None


@dataclass
class GoalAutoencoderConfig:
    n_latents: int
    n_classes: int
    encoder_cfg: MLPConfig
    decoder_cfg: MLPConfig


@dataclass
class WorkerConfig:
    action_noise: bool
    slow_target_mix: float
    extrinsic_reward: bool


@dataclass
class ManagerConfig:
    slow_target_mix: float
    intrinsic_reward: bool


@dataclass
class TrainingConfig:
    sequence_length: int
    batch_size: int
    train_every: int
    train_steps: int
    horizon: int
    seed_steps: int
    slow_target_update: int
    goal_duration: int
    save_every: int


@dataclass
class EnvironmentConfig:
    name: str
    num_parallel_envs: int
    env_args: Optional[dict]
    resize_obs: Optional[Tuple[int]] = None
    observation_shape: Optional[Tuple[int]] = None


@dataclass
class ExperimentConfig:
    training_cfg: TrainingConfig
    goal_vae_cfg: GoalAutoencoderConfig
    worker_cfg: WorkerConfig
    manager_cfg: ManagerConfig
    environment_cfg: EnvironmentConfig
    device: Device
    seed: Optional[int] = 609

@dataclass
class MultiAgent():
    '''default HPs that are known to work for MiniGrid envs'''
    #env desc
    # env : str                                           
    # obs_shape: Tuple                                            
    # action_size: int
    pixel: bool = True
    action_repeat: int = 1
    time_limit: int = 1000
    
    #buffer desc
    capacity: int = int(1e6)
    obs_dtype: np.dtype = np.uint8
    action_dtype: np.dtype = np.float32

    #training desc
    train_steps: int = int(1e6)
    train_every: int = 5
    collect_intervals: int = 5
    batch_size: int = 16
    seq_len: int = 16
    eval_episode: int = 5
    eval_render: bool = False
    save_every: int = int(5e4)
    seed_episodes: int = 5
    seed_steps: int = 4000
    model_dir: int = 'results'
    gif_dir: int = 'results'

    #latent space desc
    rssm_type: str = 'discrete'
    embedding_size: int = 100
    rssm_node_size: int = 100
    rssm_info: Dict = field(default_factory=lambda:{'deter_size':100, 'stoch_size':256, 'class_size':16, 'category_size':16, 'min_std':0.1})

    #objective desc
    grad_clip: float = 100.0
    discount_: float = 0.99
    lambda_: float = 0.95
    horizon: int = 16
    lr: Dict = field(default_factory=lambda:{'model':2e-4, 'actor':4e-5, 'critic':1e-4})
    loss_scale: Dict = field(default_factory=lambda:{'kl':1, 'reward':1.0, 'discount':10.0})
    kl: Dict = field(default_factory=lambda:{'use_kl_balance':True, 'kl_balance_scale':0.8, 'use_free_nats':False, 'free_nats':0.0})
    use_slow_target: float = True
    slow_target_update: int = 50
    slow_target_fraction: float = 1.0

    #actor critic
    actor: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'one_hot', 'min_std':1e-4, 'init_std':5, 'mean_scale':5, 'activation':nn.ELU})
    critic: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist': 'normal', 'activation':nn.ELU})
    expl: Dict = field(default_factory=lambda:{'train_noise':0.4, 'eval_noise':0.0, 'expl_min':0.05, 'expl_decay':10000.0, 'expl_type':'epsilon_greedy'})
    actor_grad: str ='reinforce'
    actor_grad_mix: int = 0.0
    actor_entropy_scale: float = 1e-3

    #learnt world-models desc
    obs_encoder: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist': None, 'activation':nn.ELU, 'kernel':2, 'depth':16})
    obs_decoder: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'normal', 'activation':nn.ELU, 'kernel':2, 'depth':16})
    reward: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'normal', 'activation':nn.ELU})
    discount: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'binary', 'activation':nn.ELU, 'use':True})

    # manager_cfg
    manager_cfg: Dict = field(default_factory=lambda:{
        'slow_target_mix': 0.001,
        'intrinsic_reward': False
    })

    # worker_cfg
    worker_cfg: Dict = field(default_factory=lambda:{
        'action_noise': False,
        'slow_target_mix': 0.001,
        'extrinsic_reward': True
    })
 
    # goal_vae_cfg
    goal_vae_cfg: Dict = field(default_factory=lambda:{
            'n_latents': 8, 
            'n_classes': 8, 
            'dist': None, 
            'activation':nn.ELU, 
            'kernel':2, 
            'depth':16
            })

     # encoder decoder config
    encoder_cfg: Dict = field(default_factory=lambda:{
        'n_layers': 3,
        'layer_size': 512

    })
    decoder_cfg: Dict = field(default_factory=lambda:{
        'n_layers': 3,
        'layer_size': 512
    })


