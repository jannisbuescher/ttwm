import jax
import jax.numpy as jnp

import flax.linen as nn

from typing import Dict, List, Sequence



class WorldModelTransformerAR(nn.Module):
    num_layers: int
    layer_width: int
    map_obs_shape: Sequence[int]

    def setup(self):
        self.token_up = {
            **{k: nn.Dense(self.layer_width) for k in obs_dims},
            "action": nn.Dense(self.layer_width),
        }
        self.token_down = {k: nn.Dense(v) for k, v in obs_dims.items()}
        self.transformer = Transformer(
            num_layers=self.num_layers,
            dim=self.layer_width,
            num_heads=4,
            dropout=0.0,
            use_causal_mask=True,
        )

        n_map_cells = self.map_obs_shape[0] * self.map_obs_shape[1]
        self.N = (
            2 * n_map_cells + 1 + 4 + 12 + 1 + 1 + 1 - 1
        )  # 1 - Number of obs and action tokens in the sequence
        self.boundaries = (
            0,
            n_map_cells,
            2 * n_map_cells,
            2 * n_map_cells + 1,
            2 * n_map_cells + 5,
            2 * n_map_cells + 17,
            2 * n_map_cells + 18,
            2 * n_map_cells + 19,
        )

    def tokenize(self, obs, action):
        batch_size = obs["block_map"].shape[0]
        # Reshape each observation component to (batch_size, sequence_length, feature_dim)
        obs_reshaped = {
            k: obs[k].reshape((batch_size, -1, obs[k].shape[-1])) for k in obs_dims
        }
        action_reshaped = action.reshape((batch_size, 1, action.shape[-1]))

        # Tokenize and concatenate
        tokens = {k: self.token_up[k](obs_reshaped[k]) for k in obs_dims}
        tokens["action"] = self.token_up["action"](action_reshaped)
        seq = jnp.concatenate([tokens[k] for k in obs_key_order + ("action",)], axis=1)

        return seq

    def detokenize(self, seq, obs):
        next_obs_pred = {
            k: seq[:, self.boundaries[i] : self.boundaries[i + 1]]
            for i, k in enumerate(obs_key_order)
        }

        for k in obs_key_order:
            x = jax.nn.softmax(self.token_down[k](next_obs_pred[k]), axis=-1)
            next_obs_pred[k] = x.reshape(obs[k].shape)

        return next_obs_pred

    def convert_obs_seq_1h(self, obs_seq, obs):
        next_obs_pred = {
            k: obs_seq[:, self.boundaries[i] : self.boundaries[i + 1]]
            for i, k in enumerate(obs_key_order)
        }

        for k in obs_key_order:
            x = jax.nn.one_hot(next_obs_pred[k], axis=-1, num_classes=obs_dims[k])
            next_obs_pred[k] = x.reshape(obs[k].shape)

        return next_obs_pred

    def __call__(self, obs, action, next_obs):
        batch_size = obs["block_map"].shape[0]

        # Tokenize
        seq = self.tokenize(obs, action)

        # Next state prediction with teacher forcing
        next_obs_reshaped = {
            k: next_obs[k].reshape((batch_size, -1, next_obs[k].shape[-1]))
            for k in obs_dims
        }
        next_obs_tokens = {k: self.token_up[k](next_obs_reshaped[k]) for k in obs_dims}
        next_obs_seq = jnp.concatenate(
            [next_obs_tokens[k] for k in obs_key_order], axis=1
        )

        # Append next observation and remove last token (end of next_obs)
        seq = jnp.concatenate([seq, next_obs_seq], axis=1)
        seq = seq[:, :-1]
        seq = self.transformer(seq)

        seq = seq[:, -self.N :]

        # Detokenize
        next_obs_pred = self.detokenize(seq, obs)
        return next_obs_pred

    @nn.compact
    def sample(self, rng, obs, action):
        batch_size = obs["block_map"].shape[0]

        # Tokenize
        seq = self.tokenize(obs, action)

        # Autoregressive sampling
        sampler = nn.scan(
            SamplingStep,
            variable_broadcast="params",
            split_rngs={"params": False},
        )(
            transformer=self.transformer,
            token_up=self.token_up,
            token_down=self.token_down,
            obs_key_order=self.obs_key_order,
            obs_dims=self.obs_dims,
            max_obs_dim=self.max_obs_dim,
            boundaries=self.boundaries,
            n_tokens_per_obs=self.N,
            temperature=1.0,
        )

        seq = jnp.concatenate(
            [seq, jnp.zeros((batch_size, self.N, self.layer_width))], axis=1
        )

        rng, _rng = jax.random.split(rng)
        _rngs = jax.random.split(_rng, self.N)

        seq, (next_obs_seq, entropies, dist) = sampler(
            seq, (jnp.arange(self.N) + self.N, _rngs)
        )
        # seq_index, batch --> batch, seq_index
        next_obs_seq = jnp.transpose(next_obs_seq, (1, 0))

        next_obs_pred = self.convert_obs_seq_1h(next_obs_seq, obs)

        return next_obs_pred, (entropies, dist)


class SamplingStep(nn.Module):
    transformer: nn.Module
    token_up: nn.Module
    token_down: nn.Module
    obs_key_order: Dict
    obs_dims: Dict
    max_obs_dim: int
    boundaries: Tuple
    n_tokens_per_obs: int
    temperature: float
    calculate_entropy: bool = True
    return_distributions: bool = True

    def token_index_to_obs_key_index(self, token_index):
        obs_key_index = 0

        token_index -= self.n_tokens_per_obs

        for i in range(len(obs_key_order)):
            obs_key_index = jax.lax.select(
                token_index < self.boundaries[-i - 1],
                len(obs_key_order) - i - 1,
                obs_key_index,
            )
        return obs_key_index

    @nn.compact
    def __call__(self, seq, x):
        index, rng = x
        # batch, token_index, token_dim
        token = self.transformer(seq)[:, index, :]
        token = token[:, None, :]

        batch_size = token.shape[0]

        obs_key_index = self.token_index_to_obs_key_index(index)

        collapsed_token = token
        collapsed_sample = jnp.zeros((batch_size, 1), dtype=jnp.int32)
        collapsed_entropy = jnp.zeros((batch_size, 1), dtype=jnp.float32)
        collapsed_dist = (
            jnp.ones((batch_size, 1, self.max_obs_dim), dtype=jnp.float32) * -jnp.inf
        )

        for i in range(len(obs_key_order)):
            k = obs_key_order[i]

            dist = self.token_down[k](token)
            if self.temperature == 0.0:
                sample = dist.argmax(axis=2)
            else:
                rng, _rng = jax.random.split(rng)
                sample = jax.random.categorical(_rng, dist / self.temperature, axis=2)
            sample_1h = jax.nn.one_hot(sample, num_classes=obs_dims[k])
            token_up = self.token_up[k](sample_1h)
            collapsed_token = jax.lax.select(
                obs_key_index == i, token_up, collapsed_token
            )
            collapsed_sample = jax.lax.select(
                obs_key_index == i, sample, collapsed_sample
            )

            if self.calculate_entropy:
                dist_norm = jax.nn.softmax(dist, axis=2)
                entropy = -jnp.sum(dist_norm * jnp.log(dist_norm), axis=2)

                collapsed_entropy = jax.lax.select(
                    obs_key_index == i, entropy, collapsed_entropy
                )

            if self.return_distributions:
                # Pad all the distributions with zeros, so we can stack them
                padded_dist = collapsed_dist.at[
                    : dist.shape[0], :, : dist.shape[2]
                ].set(dist)

                collapsed_dist = jax.lax.select(
                    obs_key_index == i, padded_dist, collapsed_dist
                )

        collapsed_token = jax.lax.stop_gradient(collapsed_token)

        collapsed_token = collapsed_token[:, 0, :]
        seq = seq.at[:, index + 1, :].set(collapsed_token)

        return jax.lax.stop_gradient(seq), (
            collapsed_sample[:, 0],
            collapsed_entropy[:, 0],
            collapsed_dist[:, 0, :],
        )