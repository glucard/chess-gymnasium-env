import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import torch.multiprocessing as mp
from queue import Empty

# ==============================================================================
# 1. ACTION PREDICTOR MODEL (Adapted for ChessEnv)
# ==============================================================================

class ChessActionPredictor(nn.Module):
    """
    An MLP that tries to predict the agent's next action (from 64*64 possibilities)
    given the current observation (an 8x8x12 tensor).
    
    This is the "behavior model" M.
    """
    def __init__(self, obs_shape, act_dim, lr=1e-4):
        super().__init__()
        self.lr = lr
        
        # Calculate the flattened observation dimension from the shape (e.g., 8*8*12)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),           # Flattens (N, 8, 8, 12) to (N, 768)
        )
        with torch.no_grad():
            # Create a dummy tensor matching the (N, C, H, W) format
            dummy_input = torch.rand((1, obs_shape[2], obs_shape[0], obs_shape[1]))
            # Pass it through the feature extractor to get the output shape
            flat_obs_dim = self.feature_extractor(dummy_input).shape[1]
            
        print(f"ChessActionPredictor feature extractor flat dim = {flat_obs_dim}")
            
        self.model = nn.Sequential(
            nn.Linear(flat_obs_dim, 256), # Increased size for complex state
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim) # Outputs logits for each 4096 action
        )        
        # CrossEntropyLoss is ideal for multi-class classification (predicting the action)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = None

    def get_prediction_error(self, obs_tensor, actual_action_tensor):
        """
        Calculates the "surprise" (our R_I) *before* training.
        This is the loss. A high loss = high surprise = high reward.
        
        obs_tensor shape: (1, 8, 8, 12)
        actual_action_tensor shape: (1)
        """
        with torch.no_grad():
            pred_logits = self.forward(obs_tensor) # Shape: (1, 4096)
            
            # Ensure action tensor is correctly shaped for CrossEntropyLoss
            # It expects (N) for targets, where N is batch size.
            if actual_action_tensor.dim() == 0:
                actual_action_tensor = actual_action_tensor.unsqueeze(0) # Shape: (1)
            
            error = self.loss_fn(pred_logits, actual_action_tensor)
        return error.item()
    

    def init_optimizer(self):
        """
        Creates the optimizer. Must be called AFTER .share_memory().
        """
        if self.optimizer is None:
            # Now, self.parameters() points to the shared memory
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
            print("Optimizer initialized and linked to shared memory.")


    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.feature_extractor(x)
        x = self.model(x)
        return x

    def train_model(self, obs_tensor, actual_action_tensor):
        """
        Trains the model to get better at predicting the action.
        As this model gets better, R_I will drop for this state-action.
        """
        if actual_action_tensor.dim() == 0:
            actual_action_tensor = actual_action_tensor.unsqueeze(0)
            
        pred_logits = self.forward(obs_tensor)

        loss = self.loss_fn(pred_logits, actual_action_tensor)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

# ==============================================================================
# 2. CUSTOM GYMNASIUM WRAPPER (Adapted for ChessEnv)
# ==============================================================================

class ChessIntrinsicRewardWrapper(gym.Wrapper):
    """
    This wrapper:
    1. Holds the ChessActionPredictor model.
    2. Calculates R_I (prediction error) at each step.
    3. Trains the ChessActionPredictor model.
    4. Adds R_I to the environment's extrinsic reward.
    """
    def __init__(self, env, chess_action_predictor:ChessActionPredictor, shared_state, beta=0.1, train_freq=50, device="cuda:0"):
        super().__init__(env)
        
        # Get observation and action space dimensions
        assert isinstance(env.observation_space, gym.spaces.Box), \
            "Wrapper requires a Box observation space."
        assert isinstance(env.action_space, gym.spaces.Discrete), \
            "Wrapper requires a Discrete action space."
        
        self.device = device
        self.local_predictor = chess_action_predictor
        self.local_predictor.to(device)
        self.local_predictor_version = -1
        self.shared_state = shared_state
        self.beta = beta  # Hyperparameter to scale R_I
        self.last_obs = None

        self.train_freq = train_freq # How often to train (in steps)
        self.step_count = 0          # Counter

    def step(self, action):
        
        # if self.step_count % self.train_freq == 0: # read memory shared
        latest_version = self.shared_state['version']

        if latest_version > self.local_predictor_version:
            print(f"ENV {os.getpid()}: Updating to version {latest_version}") # Optional: for debugging
            self.local_predictor.load_state_dict(self.shared_state['weights'])
            self.local_predictor_version = latest_version
            
        self.step_count += 1
        # 1. Get tensors for the *previous* state and *current* action
        # The observation is int8, model expects float32
        last_obs_tensor = torch.tensor(self.last_obs, dtype=torch.float32, device=self.device).unsqueeze(0) # Shape: (1, 8, 8, 12)
        action_tensor = torch.tensor(action, dtype=torch.long, device=self.device) # Shape: ()

        # 2. Calculate intrinsic reward (prediction error)
        # This is the "surprise" *before* we train the model on this step
        r_intrinsic = self.local_predictor.get_prediction_error(last_obs_tensor, action_tensor)
        
        # # 3. Train the predictor model to get better
        # if self.step_count % self.train_freq == 0:
        #     self.predictor.train_model(last_obs_tensor, action_tensor)
        
        # 4. Take the actual step in the environment
        obs, r_extrinsic, done, truncated, info = self.env.step(action)
        
        # 5. Store the new obs for the *next* step's prediction
        self.last_obs = obs
        
        # 6. Combine rewards
        r_total = r_extrinsic + (self.beta * r_intrinsic)

        # Store r_i for logging
        info['reward/raw_intrinsic'] = float(r_intrinsic)
        info['reward/intrinsic'] = float(self.beta * r_intrinsic)
        info['reward/extrinsic'] = float(r_extrinsic)
        info['reward/total'] = float(r_total)
        
        return obs, r_total, done, truncated, info

    def reset(self, **kwargs):
        # 1. Reset the underlying environment
        # This matches the ChessEnv's return signature
        obs, info = self.env.reset(**kwargs) 
        
        # 2. Store the initial obs for the first step
        self.last_obs = obs 
        
        # 3. Return the initial obs and info
        return obs, info