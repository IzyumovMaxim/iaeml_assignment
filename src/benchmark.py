import time

import equinox
import jax
import jax.numpy as jnp

from algorithms.ppo import PPOConfig, make_env
from env.base import BaseEnv
from env.robotaxi import RoboTaxiEnv

cfg = PPOConfig()
env_params, init_state = make_env(cfg)
NUM_STEPS = 128


def python_loop(n_steps=300):
    key = jax.random.PRNGKey(0)
    _, state = RoboTaxiEnv.reset(key, env_params, init_state)
    action = jnp.array([0.2, 0.0])
    RoboTaxiEnv.step(key, state, action, env_params)
    start = time.perf_counter()
    for _ in range(n_steps):
        _, state, _, done, _ = RoboTaxiEnv.step(key, state, action, env_params)
        if bool(done):
            _, state = RoboTaxiEnv.reset(key, env_params, init_state)
    return n_steps / (time.perf_counter() - start)


def make_rollout(num_envs, replan_every_step):
    def env_step(state, key):
        obs, state, reward, done, _ = RoboTaxiEnv.step(key, state, jnp.array([0.2, 0.0]), env_params)
        if replan_every_step:
            obstacles = jnp.concatenate([state.static_obstacles, state.kinematic_obstacles])
            path = BaseEnv._find_path(state.agent_pos, state.goal_pos, obstacles, env_params)
            state = equinox.tree_at(lambda s: s.path_array, state, path)
        _, reset_state = RoboTaxiEnv.reset(key, env_params, init_state)
        state = jax.tree.map(lambda r, s: jnp.where(done, r, s), reset_state, state)
        return state, reward

    def rollout(states, key):
        def body(states, key):
            return jax.vmap(env_step)(states, jax.random.split(key, num_envs))
        return jax.lax.scan(body, states, jax.random.split(key, NUM_STEPS))

    return jax.jit(rollout)


def scanned(num_envs, replan_every_step=False, repeats=5):
    keys = jax.random.split(jax.random.PRNGKey(0), num_envs)
    _, states = jax.vmap(lambda k: RoboTaxiEnv.reset(k, env_params, init_state))(keys)
    rollout = make_rollout(num_envs, replan_every_step)
    jax.block_until_ready(rollout(states, jax.random.PRNGKey(1)))
    start = time.perf_counter()
    for i in range(repeats):
        states, rewards = rollout(states, jax.random.PRNGKey(i))
    jax.block_until_ready(rewards)
    return repeats * NUM_STEPS * num_envs / (time.perf_counter() - start)


if __name__ == "__main__":
    print(f"device: {jax.devices()[0]}")
    print(f"python loop, 1 env (jitted step):      {python_loop():10.0f} steps/s")
    for n in (1, 16, 128, 512):
        print(f"jit+vmap+scan, {n:4d} envs:              {scanned(n):10.0f} steps/s")
    print(f"jit+vmap+scan, 128 envs, replan/step:  {scanned(128, True, repeats=2):10.0f} steps/s")
