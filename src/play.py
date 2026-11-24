import argparse
import os

import jax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=["keyboard", "pursuit", "ppo"], default="keyboard")
    parser.add_argument("--checkpoint", default="checkpoints/ppo_map0.eqx")
    parser.add_argument("--map-id", type=int, default=None, help="defaults to the checkpoint's map or 0")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=450)
    parser.add_argument("--record", default=None, help="output .mp4/.gif; renders headless")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.record:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    from algorithms.ppo import load_checkpoint, make_env
    from env.robotaxi import RoboTaxiEnv
    from utils.robotaxi_frontend import RoboTaxiFrontend, ppo_policy, pure_pursuit

    if args.controller == "ppo":
        model, cfg = load_checkpoint(args.checkpoint)
        if args.map_id is not None:
            cfg = cfg.__class__(**{**cfg.__dict__, "map_id": args.map_id})
        env_params, init_state = make_env(cfg)
        agent_fn = ppo_policy(model)
    else:
        env_params, init_state = RoboTaxiEnv.init_params(
            jax.random.PRNGKey(args.seed),
            map_id=args.map_id or 0,
            max_steps=args.max_steps,
            path_length=100,
            num_ray_sensors=16,
            fps=args.fps,
        )
        agent_fn = pure_pursuit() if args.controller == "pursuit" else None

    frontend = RoboTaxiFrontend(RoboTaxiEnv, env_params, init_state, agent_fn=agent_fn, seed=args.seed)
    if args.record:
        os.makedirs(os.path.dirname(args.record) or ".", exist_ok=True)
        episodes = frontend.record(args.record, num_episodes=args.episodes)
        print(f"saved {args.record}; episodes: {episodes}")
    else:
        frontend.run()


if __name__ == "__main__":
    main()
