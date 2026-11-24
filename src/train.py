import argparse
import csv
import os
import time
from dataclasses import fields

import equinox as eqx
import jax

from algorithms.ppo import PPOConfig, make_train, save_checkpoint


def parse_args() -> PPOConfig:
    parser = argparse.ArgumentParser()
    for f in fields(PPOConfig):
        flag = "--" + f.name.replace("_", "-")
        if f.type is bool or f.type == "bool":
            parser.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default)
        else:
            parser.add_argument(flag, type=type(f.default), default=f.default)
    parser.add_argument("--out", default="checkpoints")
    args = vars(parser.parse_args())
    out = args.pop("out")
    return PPOConfig(**args), out


def main():
    cfg, out_dir = parse_args()
    os.makedirs(out_dir, exist_ok=True)
    name = f"ppo_map{cfg.map_id}"
    print(cfg)
    print(f"devices: {jax.devices()}, updates: {cfg.num_updates}")

    init, update = make_train(cfg)
    runner, static = init(jax.random.PRNGKey(cfg.seed))

    log_path = os.path.join(out_dir, f"{name}_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = None
        start = time.time()
        for i in range(cfg.num_updates):
            runner, metrics = update(runner, static)
            metrics = {k: float(v) for k, v in metrics.items()}
            steps = (i + 1) * cfg.num_envs * cfg.num_steps
            row = {"update": i + 1, "timesteps": steps, "wall_time": time.time() - start, **metrics}
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            if (i + 1) % 10 == 0 or i == 0:
                f.flush()
                print(
                    f"upd {i + 1:4d} | steps {steps:9d} | {steps / row['wall_time']:7.0f} sps | "
                    f"return {metrics['ep_return']:7.2f} | len {metrics['ep_length']:6.1f} | "
                    f"success {metrics['success']:.2f} | crash {metrics['crash']:.2f} | "
                    f"kl {metrics['approx_kl']:.4f}",
                    flush=True,
                )

    model = eqx.combine(runner[0], static)
    ckpt = os.path.join(out_dir, f"{name}.eqx")
    save_checkpoint(ckpt, model, cfg)
    print(f"saved {ckpt} and {log_path}")


if __name__ == "__main__":
    main()
