import math
import shutil
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pygame

from env.robotaxi import A_MAX, DELTA_MAX, L_F, L_R
from utils.renderer import PygameFrontend


class RoboTaxiFrontend(PygameFrontend):

    def __init__(self, env, env_params, init_state, agent_fn=None, seed=0):
        super().__init__(env, env_params, init_state, eval_mode=agent_fn is not None, agent_fn=agent_fn)
        self.key = jax.random.PRNGKey(seed)
        self.obs, self.state = self.env.reset(self.key, env_params, init_state)
        self.episode_return = 0.0
        self.episodes = []
        self.info = self._display_info(0.0)

    def handle_keys(self):
        keys = pygame.key.get_pressed()
        accel = float(keys[pygame.K_UP] or keys[pygame.K_w]) - float(keys[pygame.K_DOWN] or keys[pygame.K_s])
        steer = float(keys[pygame.K_LEFT] or keys[pygame.K_a]) - float(keys[pygame.K_RIGHT] or keys[pygame.K_d])
        return jnp.array([accel, steer], dtype=jnp.float32)

    def _display_info(self, reward):
        s = self.state
        return {
            "time": f"{int(s.time) / self.params.fps:5.1f} s",
            "speed": f"{float(s.velocity[0]):5.2f} m/s",
            "steer": f"{math.degrees(float(s.steer)):5.1f} deg",
            "distance_to_path": float(self.obs.distance_to_path),
            "return": f"{self.episode_return:7.2f}",
            "episode": len(self.episodes) + 1,
        }

    def draw_info(self, screen, info, **kwargs):
        panel = pygame.Surface((190, 25 * len(info) + 10), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 170))
        screen.blit(panel, (5, 5))
        super().draw_info(screen, info, **kwargs)

    def tick(self):
        if self.agent_fn is not None:
            action = self.agent_fn(self.obs, self.state)
        else:
            action = self.handle_keys()
        self.key, k_step, k_reset = jax.random.split(self.key, 3)
        self.obs, self.state, reward, done, info = self.env.step(k_step, self.state, action, self.params)
        self.episode_return += float(reward)
        self.info = self._display_info(float(reward))
        if bool(done):
            self.episodes.append(
                (self.episode_return, bool(info["reached"]), bool(info["crashed"]), int(self.state.time))
            )
            print(f"episode {len(self.episodes)}: return={self.episode_return:.2f} "
                  f"reached={bool(info['reached'])} crashed={bool(info['crashed'])} steps={int(self.state.time)}")
            self.episode_return = 0.0
            self.obs, self.state = self.env.reset(k_reset, self.params, self.init_state)
        return bool(done)

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            self.tick()
            self.draw()
            self.clock.tick(self.params.fps)
        pygame.quit()

    def record(self, path, num_episodes=1, max_frames=5000, hold_frames=10):
        frames = []
        self.draw()
        frames.append(self._frame())
        while len(self.episodes) < num_episodes and len(frames) < max_frames:
            done = self.tick()
            self.draw()
            frames.extend([self._frame()] * (hold_frames if done else 1))
        pygame.quit()
        write_video(path, frames, fps=int(self.params.fps))
        return self.episodes

    def _frame(self):
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2)).copy()


def write_video(path, frames, fps):
    if path.endswith(".gif"):
        from PIL import Image

        images = [Image.fromarray(f) for f in frames]
        images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)
        return
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found; use a .gif output path")
    h, w, _ = frames[0].shape
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", path],
        stdin=subprocess.PIPE,
    )
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()


def pure_pursuit(lookahead=1.5, v_ref=2.0, k_speed=1.5):

    def agent_fn(obs, state):
        pos = np.asarray(state.agent_pos)
        path = np.asarray(state.path_array)
        yaw = float(state.yaw)
        i = int(np.argmin(np.linalg.norm(path - pos, axis=1)))
        ahead = np.linalg.norm(path[i:] - pos, axis=1) >= lookahead
        target = path[i + int(np.argmax(ahead))] if ahead.any() else path[-1]
        d = target - pos
        local_y = -math.sin(yaw) * d[0] + math.cos(yaw) * d[1]
        ld = max(np.linalg.norm(d), 1e-3)
        steer = math.atan(2 * (L_F + L_R) * local_y / ld**2)
        v_target = min(v_ref, 0.8 * np.linalg.norm(np.asarray(state.goal_pos) - pos) + 0.5)
        accel = k_speed * (v_target - float(state.velocity[0]))
        return jnp.array([np.clip(accel / A_MAX, -1, 1), np.clip(steer / DELTA_MAX, -1, 1)])

    return agent_fn


def ppo_policy(model):
    act = jax.jit(lambda o: model.actor(o.as_vector()))
    return lambda obs, state: jnp.clip(act(obs), -1.0, 1.0)
