import jax
import jax.numpy as jnp
import numpy as np

import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import distrax

from typing import Tuple


class QuantLayer(nn.Module):
    codebook_size: int
    z_dim: int

    def setup(self):
        self.codebook = self.param('codebook', nn.initializers.normal(0.1), (self.codebook_size, self.z_dim))

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (b h w d)
        # codebook: (k d)
        b, h, w, d = x.shape
        x = jnp.reshape(x, (b*h*w, d))
        

        x_sq = jnp.sum(x ** 2, axis=1, keepdims=True)
        codebook_sq = jnp.sum(self.codebook ** 2, axis=1)
        dot_product = jnp.dot(x, self.codebook.T)

        dists = x_sq - 2 * dot_product + codebook_sq   

        idx = jnp.argmin(dists, axis=1) # (bhw 1)

        # cos_sim = jnp.einsum('nd,kd->nk', x, self.codebook)
        # idx = jnp.argmax(cos_sim, axis=-1)
        return self.decode_indices(idx, (b, h, w, d))
        
    
    def decode_indices(self, idx, shape):
        vq = self.codebook[idx]
        vq = jnp.reshape(vq, shape)
        return vq
    

class ResBlock(nn.Module):
    hid_dim: int

    @nn.compact
    def __call__(self, x: jax.Array, *, training: bool) -> jax.Array:
        x_res = nn.Conv(features=self.hid_dim, kernel_size=(3,3), strides=(1,1), padding='SAME')(x)
        x_res = nn.BatchNorm(use_running_average=not training)(x_res)
        x_res = nn.relu(x_res)
        x_res = nn.Conv(features=self.hid_dim, kernel_size=(3,3), strides=(1,1), padding='SAME')(x_res)
        x_res = nn.BatchNorm(use_running_average=not training)(x_res)
        x_res = nn.relu(x_res)
        return x + x_res
    

class CNNEnc(nn.Module):
    num_downscale: int
    num_resblocks: int
    hid_dim: int
    z_dim: int

    @nn.compact
    def __call__(self, x: jax.Array, *, training: bool) -> jax.Array:
        for _ in range(self.num_downscale):
            x = nn.Conv(features=self.hid_dim, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
            x = nn.relu(x)
            for _ in range(self.num_resblocks):
                x = ResBlock(self.hid_dim)(x, training=training)
        x = nn.Conv(features=self.z_dim, kernel_size=(3,3), strides=(1,1), padding='SAME')(x)
        return x
    
class CNNDec(nn.Module):
    num_downscale: int
    num_resblocks: int
    hid_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x:jax.Array, *, training: bool) -> jax.Array:
        for _ in range(self.num_downscale):
            x = nn.ConvTranspose(features=self.hid_dim, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
            x = nn.relu(x)
            for _ in range(self.num_resblocks):
                x = ResBlock(self.hid_dim)(x, training=training)
        x = nn.Conv(features=self.out_dim, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.sigmoid(x)
        return x



class VQVAE(nn.Module):
    codebook_size: int
    in_dim: int
    hid_dim: int
    num_downscale: int
    num_resblocks: int
    z_dim: int

    def setup(self):
        self.encoder = CNNEnc(self.num_downscale, self.num_resblocks, self.hid_dim, self.z_dim)
        self.quant = QuantLayer(self.codebook_size, self.z_dim)
        self.decoder = CNNDec(self.num_downscale, self.num_resblocks, self.hid_dim, self.in_dim)


    @nn.compact
    def __call__(self, x: jax.Array, *, training: bool) -> Tuple[jax.Array, jax.Array, jax.Array]:
        b, c, h, w = x.shape
        x = jnp.reshape(x, (b, h, w, c))
        x = self.encoder(x, training=training)

        vq = self.quant(x)
        # Trick: replace forward output with vq but let gradients pass through x
        x_recon = x + jax.lax.stop_gradient(vq - x)

        x_recon = self.decoder(x_recon, training=training)

        x_recon = jnp.reshape(x_recon, (b, h, w, c))
        x_recon = jnp.reshape(x_recon, (b, c, h, w))

        vq = jnp.transpose(vq, (0, 3, 1, 2))
        x = jnp.transpose(x, (0, 3, 1, 2))

        return x_recon, vq, x
    
    def decode_from_indices(self, indices: jax.Array) -> jax.Array:
        # indices: (b, h, w)
        b, h, w, c = indices.shape
        assert c == 1
        idx = jnp.reshape(indices, (b*h*w, 1))
        vq = self.quant.decode_indices(idx, (b, h, w, self.z_dim))
        x = self.decoder(vq, training=False)
        return jnp.reshape(x, (b, -1, h, w))  # returns (b, c, h, w)
    
    def sample(self, rng, num_samples, shape):
        c, h, w = shape
        z = jax.random.randint(rng, shape=(num_samples, h//(2**self.num_downscale), w//(2**self.num_downscale), 1), minval=0, maxval=self.codebook_size)
        return self.decode_from_indices(z)
    

def loss_fn(x_recon: jax.Array, 
         x: jax.Array, 
         vq: jax.Array, 
         x_intermediate: jax.Array, 
         beta: float = 0.25
):
    recon_loss = jnp.mean(optax.l2_loss(x_recon, x))
    codebook_loss = 0.5 * jnp.mean((jax.lax.stop_gradient(x_intermediate) - vq) ** 2)
    commitment_loss = jnp.mean((x_intermediate - jax.lax.stop_gradient(vq)) ** 2)
    return recon_loss + codebook_loss + beta * commitment_loss

