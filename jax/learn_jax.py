import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
from functools import partial

from typing import NamedTuple


K = 10
MAX_EPISODES = 100_000
TOTAL_TIMESTEPS = 10_000
EPSILON = 0.1

class BanditState(NamedTuple):
    q_true: jnp.ndarray
    q_values: jnp.ndarray
    n_actions: jnp.ndarray
    reward_sum: float
    step_count: int
    half_steps: int
    key_timestep: jnp.ndarray


def predict(key: jnp.ndarray, q_values: jnp.ndarray):
    max_value = q_values.max()
    is_max = (q_values == max_value).astype(jnp.float32)
    probs = is_max / jnp.sum(is_max)
    return jax.random.choice(key, a=q_values.shape[0], p=probs)


def update_simple_average_bandit(bandit_state: BanditState, pred_action: jnp.ndarray, reward: jnp.ndarray):
    new_n_actions = bandit_state.n_actions.at[pred_action].add(1)

    q_values = bandit_state.q_values

    q_value_offset = (1 / new_n_actions[pred_action]) * (reward - q_values[pred_action])
    
    new_q_values = q_values.at[pred_action].add(q_value_offset)

    current_reward_to_add = jnp.where(bandit_state.step_count > bandit_state.half_steps, reward, 0)
    
    return bandit_state._replace(
        n_actions = new_n_actions,
        q_values = new_q_values,
        reward_sum = bandit_state.reward_sum + current_reward_to_add,
        step_count = bandit_state.step_count + 1
    )


def step(
    k: int,
    bandit_state: BanditState,
    _,
) -> tuple[BanditState, None]:
    
    key_next_timestep, key_tie_action, key_epsilon, key_explorer_action, key_reward_noise = jax.random.split(bandit_state.key_timestep, 5)
    
    epsilon = 0.1
    explore = jax.random.uniform(key_epsilon, shape=()) <= epsilon
    explore_fn = lambda: jax.random.randint(key_explorer_action, shape=(), minval=0, maxval=k)
    exploit_fn = lambda: predict(key_tie_action, bandit_state.q_values)
    pred_action = jax.lax.cond(explore, explore_fn, exploit_fn)
    reward = bandit_state.q_true[pred_action] + jax.random.uniform(key_reward_noise, shape=(), dtype=jnp.float32)
    
    new_bandit_state = update_simple_average_bandit(bandit_state, pred_action=pred_action, reward=reward)

    return new_bandit_state._replace(key_timestep=key_next_timestep), None

def run_episode(
    key_episode: jnp.ndarray,
    k: int,
    total_timesteps: int,

):
    all_keys = jax.random.split(key_episode, 3)
    key_episode = all_keys[0]
    key_init_q_true = all_keys[1]
    key_timestep = all_keys[2]

    q_true = 0 + 1 * jax.random.normal(key=key_init_q_true, shape=(k,))
    q_values = jnp.zeros(k, dtype=jnp.float32)
    n_actions = jnp.zeros(k, dtype=jnp.int32)

    bandit_initial_state = BanditState(
        q_true=q_true,
        q_values=q_values,
        n_actions=n_actions,
        reward_sum=0,
        step_count=0,
        half_steps=total_timesteps//2,
        key_timestep=key_timestep
    )

    step_partial_fn = partial(step, k)
    final_state, _ = jax.lax.scan(step_partial_fn, bandit_initial_state, None, total_timesteps)

    return final_state.reward_sum / final_state.step_count


run_parallel_episodes = jax.jit(
    jax.vmap(
        run_episode,
        in_axes=(0, None, None)
    ),
    static_argnums=(1, 2),
)

master_key = jax.random.PRNGKey(42)
key_episodes = jax.random.split(master_key, MAX_EPISODES)

all_rewards = run_parallel_episodes(key_episodes, K, TOTAL_TIMESTEPS)
print("all_rewards_avg:", all_rewards.mean())
hist, bins = np.histogram(all_rewards, bins=100)
plt.hist(all_rewards)
plt.show()