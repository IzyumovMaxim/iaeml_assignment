import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(log_path, out_path):
    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    steps = [int(r["timesteps"]) / 1e6 for r in rows]
    panels = [("ep_return", "episode return"), ("success", "success rate"),
              ("crash", "crash rate"), ("ep_length", "episode length, steps")]
    fig, axes = plt.subplots(1, len(panels), figsize=(16, 3.5))
    for ax, (key, title) in zip(axes, panels):
        ax.plot(steps, [float(r[key]) for r in rows])
        ax.set_title(title)
        ax.set_xlabel("env steps, M")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
