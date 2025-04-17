import jax
import jax.numpy as jnp
from typing import Any, Dict

from flax import struct
import jax
import jax.numpy as jnp
from typing import Tuple

@struct.dataclass
class TrajectoryBuffer:
    obs: jnp.ndarray              # [T, L, *obs_shape]
    actions: jnp.ndarray          # [T, L, *act_shape]
    rewards: jnp.ndarray          # [T, L]
    dones: jnp.ndarray            # [T, L]
    traj_lens: jnp.ndarray        # [T]
    max_len: int
    current_idx: int             # scalar

def create_buffer(max_trajectories: int, max_len: int, obs_shape, act_shape) -> TrajectoryBuffer:
    return TrajectoryBuffer(
        obs=jnp.zeros((max_trajectories, max_len, *obs_shape)),
        actions=jnp.zeros((max_trajectories, max_len, *act_shape)),
        rewards=jnp.zeros((max_trajectories, max_len)),
        dones=jnp.zeros((max_trajectories, max_len), dtype=bool),
        traj_lens=jnp.zeros((max_trajectories,), dtype=jnp.int32),
        max_len=max_len,
        current_idx=0
    )

def add_step(buffer: TrajectoryBuffer,
             obs: jnp.ndarray,
             action: jnp.ndarray,
             reward: float,
             done: bool) -> TrajectoryBuffer:
    i = buffer.current_idx
    t = buffer.traj_lens[i]

    def update_buffer():
        return buffer.replace(
            obs=buffer.obs.at[i, t].set(obs),
            actions=buffer.actions.at[i, t].set(action),
            rewards=buffer.rewards.at[i, t].set(reward),
            dones=buffer.dones.at[i, t].set(done),
            traj_lens=buffer.traj_lens.at[i].set(t + 1)
        )

    buffer = jax.lax.cond(
        t < buffer.max_len,
        update_buffer,
        lambda: buffer
    )

    buffer = jax.lax.cond(
        done,
        lambda: buffer.replace(current_idx=i + 1),
        lambda: buffer
    )

    return buffer

def get_batch(buffer: TrajectoryBuffer, batch_size: int, rng_key: jax.Array):
    total = buffer.current_idx
    obs = buffer.obs[:total]
    actions = buffer.actions[:total]
    rewards = buffer.rewards[:total]
    dones = buffer.dones[:total]

    # Flatten across time
    batch = (obs, actions, rewards, dones)
    batch = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), batch)
    
    total_steps = batch[0].shape[0]
    permutation = jax.random.permutation(rng_key, total_steps)

    shuffled = jax.tree.map(
        lambda x: jnp.take(x, permutation, axis=0), 
        batch
    )

    num_minibatches = total_steps // batch_size
    truncated = jax.tree.map(
        lambda x: x[:num_minibatches * batch_size], 
        shuffled
    )

    minibatches = jax.tree.map(
        lambda x: x.reshape((num_minibatches, batch_size) + x.shape[1:]),
        truncated
    )

    return minibatches 


def get_obs_a_next_batch(
    buffer: TrajectoryBuffer,
    batch_size: int,
    rng_key: jax.Array
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    total = buffer.current_idx
    obs = buffer.obs[:total]
    actions = buffer.actions[:total]
    traj_lens = buffer.traj_lens[:total]

    # Create next_obs by shifting obs
    next_obs = jnp.roll(obs, shift=-1, axis=1)

    # Mask out invalid transitions (last step of each trajectory)
    time_idx = jnp.arange(buffer.max_len)[None, :]
    valid_mask = time_idx < (traj_lens[:, None] - 1)

    obs = obs[:, :-1]
    actions = actions[:, :-1]
    next_obs = next_obs[:, :-1]
    valid_mask = valid_mask[:, :-1]

    def flatten_and_mask(x):
        flat = x.reshape((-1,) + x.shape[2:])
        return flat[valid_mask.reshape(-1)]

    flat_obs = flatten_and_mask(obs)
    flat_actions = flatten_and_mask(actions)
    flat_next_obs = flatten_and_mask(next_obs)

    # Shuffle
    total_steps = flat_obs.shape[0]
    permutation = jax.random.permutation(rng_key, total_steps)

    flat_obs = jnp.take(flat_obs, permutation, axis=0)
    flat_actions = jnp.take(flat_actions, permutation, axis=0)
    flat_next_obs = jnp.take(flat_next_obs, permutation, axis=0)

    # Truncate to be divisible by batch_size
    num_minibatches = total_steps // batch_size
    trunc_size = num_minibatches * batch_size

    flat_obs = flat_obs[:trunc_size]
    flat_actions = flat_actions[:trunc_size]
    flat_next_obs = flat_next_obs[:trunc_size]

    # Reshape into minibatches
    obs_batches = flat_obs.reshape((num_minibatches, batch_size) + flat_obs.shape[1:])
    action_batches = flat_actions.reshape((num_minibatches, batch_size) + flat_actions.shape[1:])
    next_obs_batches = flat_next_obs.reshape((num_minibatches, batch_size) + flat_next_obs.shape[1:])

    return (obs_batches, action_batches, next_obs_batches)
