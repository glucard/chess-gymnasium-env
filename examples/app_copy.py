import os
import argparse
import itertools
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

import numpy as np
import torch.multiprocessing as mp

import gymnasium as gym
from stable_baselines3.common.vec_env import (
    DummyVecEnv, 
    SubprocVecEnv, 
    VecNormalize,
    VecMonitor
)
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# --- Imports for Chess Env and Wrappers ---
from chess_gymnasium_env.envs.chess import ChessEnv
# This assumes your wrapper is in this location as per your original script
from chess_gymnasium_env.wrappers.chess_intrinsic_wrapper import ChessIntrinsicRewardWrapper, ChessActionPredictor
from chessrl.cnnextractor import CustomCNNExtractor
from chessrl.resnetextractor import CustomResNetxtractor


from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

import torch as th
from queue import Empty

N_BATCHES_ERROR_PREDICTOR = 100

class RICallback(BaseCallback):
    def __init__(self, chess_actor_predictor: ChessActionPredictor, shared_state, verbose: int = 0):
        super().__init__(verbose)
        self.chess_actor_predictor = chess_actor_predictor
        self.shared_state = shared_state
        self.info_buffer = defaultdict(list)

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        for key in self.info_buffer.keys():
            self.info_buffer[key].clear()

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        if not isinstance(self.model, OnPolicyAlgorithm):
            # This should not happen with on-policy algorithms
            if self.verbose > 0:
                print("Model or Rollout Buffer not available.")
            return

        rollout_buffer: RolloutBuffer = self.model.rollout_buffer

        observations = th.tensor(rollout_buffer.observations.reshape(-1,8,8,12))
        actions = th.tensor(rollout_buffer.actions.reshape(-1), dtype=th.long)

        losses = []

        n_batches = N_BATCHES_ERROR_PREDICTOR
        batch_size = observations.shape[0]//n_batches
        for i in range(n_batches):
            batch_obs = observations[i*batch_size:(i+1)*batch_size]
            batch_act = actions[i*batch_size:(i+1)*batch_size]
            loss = self.chess_actor_predictor.train_model(batch_obs, batch_act)
            
            losses.append(loss)

        new_weights = self.chess_actor_predictor.cpu().state_dict()
        new_version = self.shared_state['version'] + 1

        try:
            self.shared_state['weights'] = new_weights
            self.shared_state['version'] = new_version
        except Exception as e:
            print(f"Callback failed to update shared state: {e}")
        
        for k, v in self.info_buffer.items():
            if len(v) == 0:
                continue
            if isinstance(v[0], int) or isinstance(v[0], float):
                avg_v = np.mean(v)
                self.logger.record(f"{k}_mean", avg_v)
            
            if k == "win":
                print(k, v)
            
        self.logger.record("train/action_predictor_loss", sum(losses) / len(losses))

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        for env_info in self.locals["infos"]:
            for k, v in env_info.items():
                self.info_buffer[k].append(v)
        
        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


def action_mask_fn(env):
    """
    Helper function to get the action mask from the base environment.
    .unwrapped digs through all wrappers (ActionMasker, IntrinsicWrapper)
    """
    return env.unwrapped._get_action_mask()

def make_env(shared_state, beta=0.1):
    """
    Factory function for creating the environment.
    This now includes the ChessIntrinsicRewardWrapper.
    """
    def _init():
        # 1. Create the base environment
        env = ChessEnv()#render_mode="human")
        
        # 2. Wrap it with the intrinsic reward calculator
        # This wrapper adds r_intrinsic to the extrinsic reward
        local_predictor = ChessActionPredictor(env.observation_space.shape, env.action_space.n, lr=0.0005) # type: ignore
        local_predictor.eval()

        env = ChessIntrinsicRewardWrapper(
            env,
            chess_action_predictor=local_predictor,
            beta=beta,
            shared_state=shared_state,
        )
        
        # 3. Wrap it with the ActionMasker for MaskablePPO
        env = ActionMasker(env, action_mask_fn)
        
        return env
    return _init


if __name__=="__main__":
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load_model', default=False, type=bool) 
    parser.add_argument('-p', '--path', default="data/ppo_mask", type=str) 
    parser.add_argument('-b', '--beta', default=0.02, type=float, help="Scaling factor for intrinsic reward")
    parser.add_argument('-n', '--n_envs', default=12, type=int, help="Number of parallel environments")
    args = parser.parse_args()
    
    with mp.Manager() as manager:
        # 1. Create a queue that is managed by the central manager
        # This proxy object IS correctly pickled

        # Create a directory for TensorBoard logs
        timestamp = datetime.now().strftime('%Y-%m-%d__%H_%M')
        log_dir = f"./tb_logs/intrinsic_{timestamp}"

        # --- Environment Setup ---
        print(f"--- Initializing {args.n_envs} environments with IntrinsicRewardWrapper (beta={args.beta}) ---")
        
        # Create a vectorized environment
        temp_env = ChessEnv()
        obs_shape = temp_env.observation_space.shape
        act_dim = temp_env.action_space.n # type: ignore
        del temp_env
        
        master_predictor = ChessActionPredictor(obs_shape, act_dim, lr=0.0001)
        master_predictor.init_optimizer()
        print(f"Predictor shared.")


        shared_state = manager.dict({
            'version': 0,
            'weights': master_predictor.state_dict() # Put initial weights
        })
        print("Manager shared dictionary created.")

        callback = RICallback(chess_actor_predictor=master_predictor, shared_state=shared_state,verbose=2)

        env = VecMonitor(SubprocVecEnv([make_env(shared_state=shared_state, beta=args.beta) for _ in range(args.n_envs)], start_method='spawn'))
        
        
        print("--- Normalizing observations (norm_obs=True) but NOT rewards (norm_reward=False) ---")
        env = VecNormalize(env, norm_obs=True, norm_reward=False)
        # -------------------

        # Setting policy
        policy_kwargs = dict(
            features_extractor_class=CustomResNetxtractor,
            features_extractor_kwargs=dict(features_dim=1024),
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=th.nn.ReLU,
        )        

        if args.load_model:
            print("Loading model...")
            model = MaskablePPO.load(args.path, env=env)
        else:
            print("Creating new model...")
            model = MaskablePPO(
                "MlpPolicy", 
                env,
                policy_kwargs=policy_kwargs,
                learning_rate=1e-5,
                n_steps=512,
                # n_steps=32,
                batch_size=64,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.01,
                clip_range=0.2,
                # target_kl=0.05,
                vf_coef=0.5,
                seed=64,
                verbose=1,
                tensorboard_log=log_dir,
            )
            
        print(model.policy)
        print(f"Model is running on device: {model.policy.device}")

        model_dir = f"data/models/intrinsic_{timestamp}"
        os.makedirs(model_dir)
        print(f"Saving models to: {model_dir}")
        print("--- Starting training... Press Ctrl+C to stop. ---")


        try:
            for t in itertools.count():
                # The 'r_intrinsic' logged by the wrapper will now appear in
                # the 'info' dict and be automatically logged to TensorBoard
                # by the VecMonitor.
                model.learn(1_000_000, reset_num_timesteps=False, callback=callback)
                model_name = f"{t}"
                model.save(os.path.join(model_dir, model_name))

        except KeyboardInterrupt:
            print("\n--- Training interrupted. Saving final model. ---")
            model.save(os.path.join(model_dir, "final_model"))
            env.save(os.path.join(model_dir, "final_vecnormalize.pkl"))
            print(f"Model and env stats saved to {model_dir}")