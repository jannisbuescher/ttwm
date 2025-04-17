import unittest

import jax
import jax.numpy as jnp
import numpy.testing as npt

from wm_utils import TrajectoryBuffer, create_buffer, add_step, get_batch, get_obs_a_next_batch

class TestTrajBuffer(unittest.TestCase):

    def test_create(self):
        buffer = create_buffer(1000, 20, (10,10), (1,))
        self.assertIsInstance(buffer, TrajectoryBuffer)

    def test_add_step(self):
        buffer = create_buffer(1000, 20, (10,10), (1,))
        in_obs = jnp.ones((10, 10))
        in_action = jnp.ones((1,))
        in_reward = 1.0
        in_done = False
        buffer = add_step(buffer, in_obs, in_action, in_reward, in_done)

        internal_state_obs = buffer.obs[0,0]
        internal_state_action = buffer.actions[0,0]
        internal_state_reward = buffer.rewards[0,0]
        internal_state_done = buffer.dones[0,0]

        npt.assert_array_equal(in_obs, internal_state_obs)
        npt.assert_array_equal(in_action, internal_state_action)
        npt.assert_array_equal(in_reward, internal_state_reward)
        npt.assert_array_equal(in_done, internal_state_done.item())
        self.assertEqual(0, buffer.current_idx)

        in_obs = jnp.ones((10, 10)) * 10
        in_action = jnp.ones((1,))* 8
        in_reward = 4.0
        in_done = True

        buffer = add_step(buffer, in_obs, in_action, in_reward, in_done)

        internal_state_obs = buffer.obs[0,1]
        internal_state_action = buffer.actions[0,1]
        internal_state_reward = buffer.rewards[0,1]
        internal_state_done = buffer.dones[0,1]

        npt.assert_array_equal(in_obs, internal_state_obs)
        npt.assert_array_equal(in_action, internal_state_action)
        npt.assert_array_equal(in_reward, internal_state_reward)
        npt.assert_array_equal(in_done, internal_state_done.item())
        self.assertEqual(1, buffer.current_idx)

    def test_get_batch_shape(self):
        buffer = create_buffer(1000, 20, (10,10), (1,))
        rng = jax.random.key(0)

        in_obs = jnp.ones((10, 10))
        in_action = jnp.ones((1,))
        in_reward = 1.0
        in_done = True
        for _ in range(100):
            buffer = add_step(buffer, in_obs, in_action, in_reward, in_done)

        batch = get_batch(buffer, 64, rng)
        obs = batch[0]
        actions = batch[1]
        rewards = batch[2]
        dones = batch[3]

        self.assertTupleEqual(obs[0].shape, (64, 10, 10))
        self.assertTupleEqual(actions[0].shape, (64, 1))
        self.assertTupleEqual(rewards[0].shape, (64,))
        self.assertTupleEqual(dones[0].shape, (64,))


    def test_get_batch_scanned(self):
        buffer = create_buffer(1000, 20, (10,10), (1,))
        rng = jax.random.key(0)

        in_obs = jnp.ones((10, 10))
        in_action = jnp.ones((1,))
        in_reward = 1.0
        in_done = False
        for _ in range(10):
            for _ in range(19):
                buffer = add_step(buffer, in_obs, in_action, in_reward, in_done)
            buffer = add_step(buffer, in_obs, in_action, in_reward, True)

        minibatches = get_obs_a_next_batch(buffer, 16, rng)

        def _update_minibatch(unused, batch_info):
            (
                obs,
                action,
                next_obs
            ) = batch_info
            self.assertTupleEqual(obs.shape, (16, 10, 10))
            self.assertTupleEqual(action.shape, (16, 1))
            self.assertTupleEqual(next_obs.shape, (16, 10, 10))
            return None, 0

        train_state, losses = jax.lax.scan(
            _update_minibatch, None, minibatches
        )
