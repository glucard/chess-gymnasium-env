import os

from chessrl.cnnextractor import CustomCNNExtractor
from chess_gymnasium_env.envs.chess import ChessEnv
import gymnasium as gym
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib import MaskablePPO

from chessrl.utils import make_env
from stable_baselines3.common.vec_env import VecMonitor

import itertools
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

if __name__=="__main__":
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load_model', default=False, type=bool) 
    parser.add_argument('-p', '--path', default="data/ppo_mask", type=str) 
    args = parser.parse_args()
    
    # Number of parallel environments
    n_envs = 2

    # Create a vectorized environment
    # env = DummyVecEnv([make_env() for i in range(n_envs)])

    # OR for SubprocVecEnv (more efficient for computationally intensive tasks)
    env = VecMonitor(SubprocVecEnv([make_env() for _ in range(n_envs)]))
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # setting policy
    policy_kwargs = dict(
        features_extractor_class=CustomCNNExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[128, 128], vf=[128, 128])  # Proper syntax
    )
    # Create a directory for TensorBoard logs
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    log_dir = f"./tb_logs/{timestamp}"
    # writer = SummaryWriter(log_dir=log_dir)

    if args.load_model:
        print("Loading model...")
        model = MaskablePPO.load("data/ppo_mask", env=env)
    else:
        model = MaskablePPO(
            "MlpPolicy", 
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=2e-5,  # Lower learning rate for smoother updates
            n_steps=256,  # Increase timesteps per update for more stable gradients
            batch_size=128,  # Increase batch size for more stable training
            gamma=0.99,  # Default gamma value for discounting
            gae_lambda=0.95,  # Default GAE Lambda for advantage estimation
            ent_coef=0.001,  # Entropy coefficient to encourage exploration
            clip_range=0.2,  # Smaller clip range for conservative updates
            target_kl=0.05,  # Allow larger policy updates before early stopping
            vf_coef=0.4,  # Lower value loss coefficient for more focus on policy updates
            seed=32,                # Random seed for reproducibility
            verbose=1,              # Set verbosity level to 1 for progress logging
            tensorboard_log=log_dir
        )
    print(model.policy)
    print(f"Model is running on device: {model.policy.device}")

    model_dir = f"data/models/{timestamp}"
    os.makedirs(model_dir)
    try:
        for t in itertools.count():
            model.learn(1_000_000, reset_num_timesteps=False)
            model_name = f"{t}"
            model.save(os.path.join(model_dir,model_name))

    except KeyboardInterrupt:
        model.save(os.path.join(model_dir,model_name))
    
    # if not os.path.isdir("data"):
    #     os.makedirs("data")
    # model.save("data/ppo_mask")
    # env.save("data/env.pkl")
