import math

from flax import linen as nn
import jax
import jax.numpy as jnp


class PositionalEncoding(nn.Module):
    """https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html"""

    d_model: int  # Hidden dimensionality of the input.
    max_len: int = 5000  # Maximum length of a sequence to expect.

    def setup(self):
        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        self.pe = jnp.zeros((self.max_len, self.d_model))
        position = jnp.arange(0, self.max_len, dtype=jnp.float32)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model)
        )
        self.pe = self.pe.at[:, 0::2].set(jnp.sin(position * div_term))
        self.pe = self.pe.at[:, 1::2].set(jnp.cos(position * div_term))

    def __call__(self, x):
        x = x + self.pe[: x.shape[-2]]
        return x


class AttentionBlock(nn.Module):
    dim: int
    num_heads: int
    dropout: float
    use_causal_mask: bool = True

    @nn.remat
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # --- Attention ---
        z = PositionalEncoding(self.dim)(x)
        z = nn.LayerNorm()(z)
        causal_mask = jnp.tri(z.shape[-2]) if self.use_causal_mask else None
        z = nn.MultiHeadAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            dropout_rate=self.dropout,
        )(z, mask=causal_mask)
        x = x + z

        # --- Feedforward ---
        z = nn.LayerNorm()(x)
        z = nn.Dense(self.dim)(z)
        z = nn.gelu(z)
        x = x + z

        return x


class Transformer(nn.Module):
    num_layers: int
    dim: int
    num_heads: int
    dropout: float = 0.0
    use_causal_mask: bool = True

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Sequential(
            [
                nn.LayerNorm(),
                nn.Dense(self.dim),
                nn.LayerNorm(),
            ]
        )(x)
        for _ in range(self.num_layers):
            x = AttentionBlock(
                dim=self.dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                use_causal_mask=self.use_causal_mask
            )(x)
        return x