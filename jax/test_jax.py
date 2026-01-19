import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
from functools import partial

# --- 1. DATA STRUCTURES (Replacing __init__) ---
# Instead of a class with 'self', we define a container for your state.
# This acts like a C struct.
from typing import NamedTuple

class BanditState(NamedTuple):
    action_values: jnp.ndarray      # Like self.action_values
    action_preferences: jnp.ndarray # For Gradient Bandit
    avg_rewards: float              # For Gradient Bandit baseline
    time: int                       # Like self.t
    
    # Environment state (The "True" values hidden in fit_testbed)
    true_bandit_rewards: jnp.ndarray 

# --- 2. LOGIC FUNCTIONS (Replacing methods) ---

# Replaces: self.predict()
def predict_gradient(key, state, k):
    # jax.nn.softmax is highly optimized for GPU
    probs = jax.nn.softmax(state.action_preferences)
    # Sampling from categorical distribution
    action = random.choice(key, k, p=probs)
    return action

# Replaces: self.update()
def update_gradient(state, action, reward, alpha, k):
    # 1. Calculate probabilities (same as predict)
    probs = jax.nn.softmax(state.action_preferences)
    
    # 2. Create the "mask" (one-hot encoding)
    # This replaces: (np.arange(self.k) == action).astype(int)
    one_hot = jax.nn.one_hot(action, k)
    
    # 3. Calculate Gradient (Your logic preserved)
    baseline = state.avg_rewards
    gradients = alpha * (reward - baseline) * (one_hot - probs)
    
    # 4. Update Preferences and Baseline
    new_preferences = state.action_preferences + gradients
    new_baseline = baseline + alpha * (reward - baseline) # Constant alpha baseline
    
    # Return a NEW state (Functional programming!)
    return state._replace(
        action_preferences=new_preferences,
        avg_rewards=new_baseline
    )

# --- 3. THE SIMULATION LOOP (Replacing fit_testbed) ---

from functools import partial # <--- Add this import

# 1. Update simulation_step to take 'k' and 'alpha' as direct arguments
#    (We removed 'params' from the tuple unpacking)
def simulation_step(k, alpha, reward_noise_std, walk_std, state, step_key):
    # Unpack ONLY the state (which changes)
    # The constants (k, alpha, etc.) are passed directly via partial
    
    # Split randomness keys
    k1, k2, k3 = random.split(step_key, 3)
    
    # A. Evolve Environment
    # Now 'k' is a real integer, so (k,) is valid!
    walk_noise = random.normal(k1, (k,)) * walk_std
    new_true_rewards = state.true_bandit_rewards + walk_noise
    
    # B. Predict
    action = predict_gradient(k2, state, k)
    
    # C. Get Reward
    reward_noise = random.normal(k3) * reward_noise_std
    reward = new_true_rewards[action] + reward_noise
    
    # D. Update Agent
    temp_state = state._replace(true_bandit_rewards=new_true_rewards)
    new_state = update_gradient(temp_state, action, reward, alpha, k)
    
    # Return (New State, Output)
    # Notice we return ONLY state as carry, not (state, params)
    return new_state, reward

# 2. Update run_single_episode to use 'partial'
def run_single_episode(key, alpha, k, timesteps):
    initial_true_rewards = jnp.zeros(k)
    
    init_state = BanditState(
        action_values=jnp.zeros(k),
        action_preferences=jnp.zeros(k),
        avg_rewards=0.0,
        time=0,
        true_bandit_rewards=initial_true_rewards
    )
    
    # Generate random keys
    step_keys = random.split(key, timesteps)
    
    # THE FIX: Use partial to freeze k, alpha, and std_devs
    # This creates a new function that only takes (state, step_key)
    # 'k' is captured as a concrete int here!
    step_fn = partial(simulation_step, k, alpha, 1.0, 0.01)
    
    # Run scan with the partial function
    # Note: carry is now just 'init_state', not (init_state, params)
    final_state, rewards_history = jax.lax.scan(
        step_fn, 
        init_state, 
        step_keys
    )
    
    last_half_rewards = rewards_history[timesteps//2:]
    return jnp.mean(last_half_rewards)

# 3. Ensure your JIT/VMap definition stays the same
def run_parallel_episodes_logic(keys, alpha, k, timesteps):
    return run_single_episode(keys, alpha, k, timesteps)

run_parallel_episodes = jax.jit(
    jax.vmap(
        run_parallel_episodes_logic, 
        in_axes=(0, None, None, None)
    ),
    static_argnums=(2, 3) # Keeps 'k' static so it can be partial()'d safely
)

# 1. The Logic for ONE Alpha (200 episodes)
# This maps over 'keys' (axis 0) but keeps 'alpha' constant
@partial(jax.jit, static_argnums=(2, 3))
@partial(jax.vmap, in_axes=(0, None, None, None)) 
def run_batch_for_alpha(keys, alpha, k, timesteps):
    return run_single_episode(keys, alpha, k, timesteps)


# 2. The Logic for ALL Alphas (8 alphas * 200 episodes)
# This maps over 'keys' (axis 0) AND 'alphas' (axis 0)
@partial(jax.jit, static_argnums=(2, 3))
@partial(jax.vmap, in_axes=(0, 0, None, None))
def run_parameter_sweep(keys_matrix, alphas_array, k, timesteps):
    return run_batch_for_alpha(keys_matrix, alphas_array, k, timesteps)

# --- EXECUTION ---

# Constants
K = 10
MAX_TIMESTEPS = 200_000 
MAX_EPISODES = 200 
ALPHAS = jnp.geomspace(1/32, 4, 8)

# Prepare Data
master_key = random.PRNGKey(42)

# We need a matrix of keys: Shape (8, 200)
# Each alpha needs its own set of 200 episode keys
master_key, *subkeys = random.split(master_key, len(ALPHAS) + 1)
keys_matrix = jnp.array([random.split(k, MAX_EPISODES) for k in subkeys])

print("Compiling and Running...")
import time
start = time.time()

# ONE CALL to rule them all
# JAX will compile one kernel to do the entire parameter study
all_rewards = run_parameter_sweep(keys_matrix, ALPHAS, K, MAX_TIMESTEPS)

# Force JAX to finish (it's asynchronous)
all_rewards.block_until_ready()

end = time.time()
print(f"Total Time: {end - start:.4f}s")

# Result shape is (8, 200) -> Average over episodes (axis 1)
avg_rewards = jnp.mean(all_rewards, axis=1)

# Print results
for alpha, reward in zip(ALPHAS, avg_rewards):
    print(f"Alpha {alpha:.4f}: {reward:.4f}")