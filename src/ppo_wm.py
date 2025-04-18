import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from craftax.craftax_env import make_craftax_env_from_name

import wandb
from typing import NamedTuple

from flax.training import orbax_utils
from flax.training.train_state import TrainState
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

from logz.batch_logging import batch_log, create_log_dict
from models.actor_critic import (
    ActorCritic,
    ActorCriticConv,
)
# from models.icm import ICMEncoder, ICMForward, ICMInverse
from wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
)

from wm_utils import TrajectoryBuffer, create_buffer
from wm_utils import add_step as buffer_add_step
from wm_utils import get_obs_a_next_batch as buffer_get_batch

from twm import WorldModelTransformerAR

# Code adapted from the original implementation made by Chris Lu
# Original code located at https://github.com/luchris429/purejaxrl


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward_e: jnp.ndarray
    reward_i: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = make_craftax_env_from_name(
        config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"]
    )
    env_params = env.default_params

    env = LogWrapper(env)
    if config["USE_OPTIMISTIC_RESETS"]:
        env = OptimisticResetVecEnvWrapper(
            env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
        )
    else:
        env = AutoResetEnvWrapper(env)
        env = BatchEnvWrapper(env, num_envs=config["NUM_ENVS"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        if "Symbolic" in config["ENV_NAME"]:
            network = ActorCriticConv(env.action_space(env_params).n, config["LAYER_SIZE"])
            world_model = WorldModelTransformerAR(
                num_layers=3,
                layer_width=100,
                map_obs_shape=(*env.observation_space(env_params).shape,),
                obs_dims=None,
                obs_key_order=None
            )

        else:
            network = ActorCriticConv(
                env.action_space(env_params).n, config["LAYER_SIZE"]
            )

        rng, _rng, _rng_wm = jax.random.split(rng, 3)
        init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
        init_wm = (
            jnp.zeros((1, *env.observation_space(env_params).shape)),
            jnp.zeros((1, env.action_space(env_params).n)),
            jnp.zeros((1, *env.observation_space(env_params).shape))
        )

        network_params = network.init(_rng, init_x)
        wm_params = world_model.init(_rng_wm, *init_wm)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            tx_wm = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            tx_wm = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        train_state_wm = TrainState.create(
            apply_fn=world_model.apply,
            params=wm_params,
            tx=tx_wm
        )

        # Exploration state
        ex_state = {
            "icm_encoder": None,
            "icm_forward": None,
            "icm_inverse": None,
            "e3b_matrix": None,
        }


        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            traj_buffer = create_buffer(
                max_trajectories=config['SIZE_TRAJ_BUFFER'],
                max_len=config["NUM_STEPS"],
                obs_shape=(*env.observation_space(env_params).shape,),
                act_shape=env.action_space(env_params).n
            )

            # COLLECT TRAJECTORIES
            def _env_step_and_collect(runner_state, unused):
                (
                    train_state,
                    train_state_wm,
                    env_state,
                    last_obs,
                    ex_state,
                    rng,
                    update_step,
                ) = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward_e, done, info = env.step(
                    _rng, env_state, action, env_params
                )

                reward_i = jnp.zeros(config["NUM_ENVS"])

                reward = reward_e + reward_i

                transition = Transition(
                    done=done,
                    action=action,
                    value=value,
                    reward=reward,
                    reward_i=reward_i,
                    reward_e=reward_e,
                    log_prob=log_prob,
                    obs=last_obs,
                    next_obs=obsv,
                    info=info,
                )
                runner_state = (
                    train_state,
                    train_state_wm,
                    env_state,
                    obsv,
                    ex_state,
                    rng,
                    update_step,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step_and_collect, runner_state, None, config['NUM_COLLECTION_STEPS']
            )

            # TRAIN WORLD MODEL
            def _update_wm_step(wm_update_state, unused):

                (
                    train_state_wm,
                    rng,
                    update_step
                ) = wm_update_state

                def _update_epoch(update_state, unused):
                    def _update_minibatch(train_state_wm, batch_info):
                        (
                            obs,
                            action,
                            next_obs
                        ) = batch_info
                        def _loss_fn(params, obs, action, next_obs):
                            next_obs_pred = world_model.apply(params, obs, action, next_obs)

                            loss = 0.0
                            for k in next_obs:
                                loss += jnp.mean((next_obs_pred[k] - next_obs[k]) ** 2)
                            return loss
                        
                        grad_fn = jax.value_and_grad(_loss_fn)
                        total_loss, grads = grad_fn(train_state_wm.params, obs, action, next_obs)
                        train_state_wm = train_state_wm.apply_gradients(grads=grads)

                        return train_state_wm, total_loss
            
                    (
                        train_state_wm,
                        rng
                    ) = update_state
                    rng, _rng = jax.random.split(rng)
                    minibatches = buffer_get_batch(traj_buffer, config['MINIBATCH_SIZE'], _rng)
                    
                    train_state_wm, losses = jax.lax.scan(
                        _update_minibatch, train_state_wm, minibatches
                    )

                    updated_state = (
                        train_state_wm,
                        rng
                    )
                    return updated_state, losses

                update_state = (
                    train_state_wm,
                    rng,
                )
                update_state, loss_info = jax.lax.scan(
                    _update_epoch, update_state, None, config['WM_TRAIN_EPOCHS']
                )

                train_state_wm = update_state[0]
                rng = update_state[-1]

                # wandb logging
                if config["DEBUG"] and config["USE_WANDB"]:

                    def callback(metric, update_step):
                        to_log = create_log_dict(metric, config)
                        batch_log(update_step, to_log, config)

                    jax.debug.callback(
                        callback,
                        loss_info,
                        update_step,
                    )
                wm_update_state = (
                    train_state_wm,
                    rng,
                    update_step
                )
                return wm_update_state

            (
                train_state,
                train_state_wm,
                env_state,
                last_obs,
                rng,
                update_step,
            ) = runner_state

            wm_update_state = (
                train_state_wm,
                rng,
                update_step
            )

            wm_update_state = _update_wm_step(wm_update_state, None)

            runner_state = (
                train_state,
                train_state_wm,
                env_state,
                last_obs,
                rng,
                update_step,
            )



            # TRAIN ACTOR/CRITIC
            def _train_step(runner_state, unused):
                def _imagine(runner_state, unused):
                    (
                        train_state,
                        train_state_wm,
                        env_state,
                        last_obs,
                        rng,
                        update_step,
                    ) = runner_state

                    # SELECT ACTION
                    rng, _rng = jax.random.split(rng)
                    pi, value = network.apply(train_state.params, last_obs)
                    action = pi.sample(seed=_rng)
                    log_prob = pi.log_prob(action)

                    # STEP ENV
                    rng, _rng = jax.random.split(rng)
                    next_obs_pred, (entropy, dist) = world_model.apply(
                        train_state_wm.params, _rng, last_obs, action, method=WorldModelTransformerAR.sample
                    )

                    obsv, env_state, reward, done, info = next_obs_pred

                    transition = Transition(
                        done=done,
                        action=action,
                        value=value,
                        reward=reward,
                        log_prob=log_prob,
                        obs=last_obs,
                        next_obs=obsv,
                        info=info,
                    )
                    runner_state = (
                        train_state,
                        train_state_wm,
                        env_state,
                        obsv,
                        rng,
                        update_step,
                    )
                    return runner_state, transition                    


                runner_state, traj_batch = jax.lax.scan(
                    _imagine, runner_state, None, config["NUM_STEPS"]
                )
                
                # CALCULATE ADVANTAGE
                (
                    train_state,
                    train_state_wm,
                    env_state,
                    last_obs,
                    ex_state,
                    rng,
                    update_step,
                ) = runner_state
                _, last_val = network.apply(train_state.params, last_obs)


                def _calculate_gae(traj_batch, last_val):
                    def _get_advantages(gae_and_next_value, transition):
                        gae, next_value = gae_and_next_value
                        done, value, reward = (
                            transition.done,
                            transition.value,
                            transition.reward,
                        )
                        delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                        gae = (
                            delta
                            + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                        )
                        return (gae, value), gae

                    _, advantages = jax.lax.scan(
                        _get_advantages,
                        (jnp.zeros_like(last_val), last_val),
                        traj_batch,
                        reverse=True,
                        unroll=16,
                    )
                    return advantages, advantages + traj_batch.value

                advantages, targets = _calculate_gae(traj_batch, last_val)

                # UPDATE NETWORK
                def _update_epoch(update_state, unused):
                    def _update_minbatch(train_state, batch_info):
                        traj_batch, advantages, targets = batch_info

                        # Policy/value network
                        def _loss_fn(params, traj_batch, gae, targets):
                            # RERUN NETWORK
                            pi, value = network.apply(params, traj_batch.obs)
                            log_prob = pi.log_prob(traj_batch.action)

                            # CALCULATE VALUE LOSS
                            value_pred_clipped = traj_batch.value + (
                                value - traj_batch.value
                            ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                            value_losses = jnp.square(value - targets)
                            value_losses_clipped = jnp.square(value_pred_clipped - targets)
                            value_loss = (
                                0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                            )

                            # CALCULATE ACTOR LOSS
                            ratio = jnp.exp(log_prob - traj_batch.log_prob)
                            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                            loss_actor1 = ratio * gae
                            loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config["CLIP_EPS"],
                                    1.0 + config["CLIP_EPS"],
                                )
                                * gae
                            )
                            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                            loss_actor = loss_actor.mean()
                            entropy = pi.entropy().mean()

                            total_loss = (
                                loss_actor
                                + config["VF_COEF"] * value_loss
                                - config["ENT_COEF"] * entropy
                            )
                            return total_loss, (value_loss, loss_actor, entropy)

                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        total_loss, grads = grad_fn(
                            train_state.params, traj_batch, advantages, targets
                        )
                        train_state = train_state.apply_gradients(grads=grads)

                        losses = (total_loss, 0)
                        return train_state, losses

                    (
                        train_state,
                        traj_batch,
                        advantages,
                        targets,
                        rng,
                    ) = update_state
                    rng, _rng = jax.random.split(rng)
                    batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                    assert (
                        batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                    ), "batch size must be equal to number of steps * number of envs"
                    permutation = jax.random.permutation(_rng, batch_size)
                    batch = (traj_batch, advantages, targets)
                    batch = jax.tree.map(
                        lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                    )
                    shuffled_batch = jax.tree.map(
                        lambda x: jnp.take(x, permutation, axis=0), batch
                    )
                    minibatches = jax.tree.map(
                        lambda x: jnp.reshape(
                            x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                        ),
                        shuffled_batch,
                    )
                    train_state, losses = jax.lax.scan(
                        _update_minbatch, train_state, minibatches
                    )
                    update_state = (
                        train_state,
                        traj_batch,
                        advantages,
                        targets,
                        rng,
                    )
                    return update_state, losses

                update_state = (
                    train_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                update_state, loss_info = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
                )

                train_state = update_state[0]
                metric = jax.tree.map(
                    lambda x: (x * traj_batch.info["returned_episode"]).sum()
                    / traj_batch.info["returned_episode"].sum(),
                    traj_batch.info,
                )

                rng = update_state[-1]

                # wandb logging
                if config["DEBUG"] and config["USE_WANDB"]:

                    def callback(metric, update_step):
                        to_log = create_log_dict(metric, config)
                        batch_log(update_step, to_log, config)

                    jax.debug.callback(
                        callback,
                        metric,
                        update_step,
                    )

                runner_state = (
                    train_state,
                    train_state_wm,
                    env_state,
                    last_obs,
                    ex_state,
                    rng,
                    update_step + 1,
                )
                return runner_state, metric
            

            runner_state, metric = jax.lax.scan(
                _train_step, runner_state, None, config['UPDATE_EPOCHS']
            )

            (
                train_state,
                train_state_wm,
                env_state,
                last_obs,
                ex_state,
                rng,
                update_step
            ) = runner_state

            runner_state = (
                train_state,
                train_state_wm,
                env_state,
                last_obs,
                ex_state,
                rng,
                update_step + 1
            )

            return runner_state, metric
                



        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            train_state_wm,
            env_state,
            obsv,
            ex_state,
            _rng,
            0,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}  # , "info": metric}

    return train


def run_ppo(config):
    config = {k.upper(): v for k, v in config.__dict__.items()}

    if config["USE_WANDB"]:
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=config["ENV_NAME"]
            + "-"
            + str(int(config["TOTAL_TIMESTEPS"] // 1e6))
            + "M",
        )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"])

    train_jit = jax.jit(make_train(config))
    train_vmap = jax.vmap(train_jit)

    t0 = time.time()
    out = train_vmap(rngs)
    t1 = time.time()
    print("Time to run experiment", t1 - t0)
    print("SPS: ", config["TOTAL_TIMESTEPS"] / (t1 - t0))

    if config["USE_WANDB"]:

        def _save_network(rs_index, dir_name):
            train_states = out["runner_state"][rs_index]
            train_state = jax.tree.map(lambda x: x[0], train_states)
            orbax_checkpointer = PyTreeCheckpointer()
            options = CheckpointManagerOptions(max_to_keep=1, create=True)
            path = os.path.join(wandb.run.dir, dir_name)
            checkpoint_manager = CheckpointManager(path, orbax_checkpointer, options)
            print(f"saved runner state to {path}")
            save_args = orbax_utils.save_args_from_target(train_state)
            checkpoint_manager.save(
                config["TOTAL_TIMESTEPS"],
                train_state,
                save_kwargs={"save_args": save_args},
            )

        if config["SAVE_POLICY"]:
            _save_network(0, "policies")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Craftax-Symbolic-v1")
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--total_timesteps", type=lambda x: int(float(x)), default=1e9
    )  # Allow scientific notation
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.8)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument(
        "--anneal_lr", action=argparse.BooleanOptionalAction, default=True
    )
    # WM additional arguments
    parser.add_argument("--size_traj_buffer", type=int, default=1e5)
    parser.add_argument("--wm_train_epochs", type=int, default=10)
    parser.add_argument("--num_collection_steps", type=int, default=1e4)

    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--use_wandb", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--save_policy", action="store_true")
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--layer_size", type=int, default=512)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument(
        "--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)

    # EXPLORATION
    parser.add_argument("--exploration_update_epochs", type=int, default=4)
    # ICM
    parser.add_argument("--icm_reward_coeff", type=float, default=1.0)
    parser.add_argument("--train_icm", action="store_true")
    parser.add_argument("--icm_lr", type=float, default=3e-4)
    parser.add_argument("--icm_forward_loss_coef", type=float, default=1.0)
    parser.add_argument("--icm_inverse_loss_coef", type=float, default=1.0)
    parser.add_argument("--icm_layer_size", type=int, default=256)
    parser.add_argument("--icm_latent_size", type=int, default=32)
    # E3B
    parser.add_argument("--e3b_reward_coeff", type=float, default=1.0)
    parser.add_argument("--use_e3b", action="store_true")
    parser.add_argument("--e3b_lambda", type=float, default=0.1)

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.use_e3b:
        assert args.train_icm
        assert args.icm_reward_coeff == 0
    if args.seed is None:
        args.seed = np.random.randint(2**31)

    if args.jit:
        run_ppo(args)
    else:
        with jax.disable_jit():
            run_ppo(args)
