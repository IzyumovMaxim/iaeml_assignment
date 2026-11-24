import argparse

import jax
import jax.numpy as jnp

from algorithms.ppo import load_checkpoint, make_env
from env.robotaxi import RoboTaxiEnv


def evaluate(model, cfg, num_episodes=256, seed=123):
    env_params, init_state = make_env(cfg)

    def episode(key):
        obs, state = RoboTaxiEnv.reset(key, env_params, init_state)

        def body(carry, key):
            obs, state, ret, done, reached, crashed, length = carry
            action = jnp.clip(model.actor(obs.as_vector()), -1, 1)
            obs, new_state, r, d, info = RoboTaxiEnv.step(key, state, action, env_params)
            alive = ~done
            carry = (
                obs, new_state, ret + r * alive, done | d,
                reached | (info["reached"] & alive), crashed | (info["crashed"] & alive),
                length + alive,
            )
            return carry, None

        init = (obs, state, 0.0, False, False, False, 0)
        (_, _, ret, _, reached, crashed, length), _ = jax.lax.scan(
            body, init, jax.random.split(key, cfg.max_steps)
        )
        return ret, reached, crashed, length

    ret, reached, crashed, length = jax.jit(jax.vmap(episode))(
        jax.random.split(jax.random.PRNGKey(seed), num_episodes)
    )
    return {
        "episodes": num_episodes,
        "return": float(ret.mean()),
        "success": float(reached.mean()),
        "crash": float(crashed.mean()),
        "timeout": float(1 - reached.mean() - crashed.mean()),
        "time_to_goal_s": float(jnp.sum(length * reached) / jnp.maximum(reached.sum(), 1) / cfg.fps),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--map-id", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=256)
    args = parser.parse_args()
    model, cfg = load_checkpoint(args.checkpoint)
    if args.map_id is not None:
        cfg = cfg.__class__(**{**cfg.__dict__, "map_id": args.map_id})
    print(f"{args.checkpoint} on map {cfg.map_id}: {evaluate(model, cfg, args.episodes)}")
