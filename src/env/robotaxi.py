import heapq
from functools import partial
from typing import Any, Dict, Tuple

import chex
import equinox
import jax
import numpy as np
from jax import numpy as jnp

from env.base import FOV, FPS, NUM_RAY_SENSORS, BaseEnv, BaseEnvParams, BaseEnvState
from env.tasks import BaseTaskSampler
from utils import utils

G = 9.81
MASS = 150.0  # kg
L_F = 0.45  # m
L_R = 0.45  # m
I_Z = 20.0  # kg m^2
C_F = 2500.0  # N/rad
C_R = 2500.0  # N/rad
MU = 0.9
FZ_F = MASS * G * L_R / (L_F + L_R)  # N
FZ_R = MASS * G * L_F / (L_F + L_R)  # N
C_ROLL = 0.02
C_DRAG = 3.0  # N s^2/m^2

A_MAX = 3.0  # m/s^2
DELTA_MAX = 0.5  # rad
STEER_TAU = 0.1  # s
STEER_RATE_MAX = 2.0  # rad/s
V_MAX = 4.0  # m/s
V_MIN = -1.5  # m/s
V_BLEND_LOW = 0.5  # m/s
V_BLEND_HIGH = 1.5  # m/s
PHYSICS_HZ = 120  # Hz

W_PROGRESS = 1.0
W_PATH = 0.02
W_TIME = 0.01
W_SMOOTH = 0.01
R_GOAL = 10.0
R_CRASH = 10.0

YAW_RESET_NOISE = 0.3  # rad
LOOKAHEAD = (3, 6)


class RoboTaxiParams(BaseEnvParams):
    physics_substeps: int


class RoboTaxiState(BaseEnvState):
    yaw: chex.Scalar
    velocity: chex.Array
    yaw_rate: chex.Scalar
    steer: chex.Scalar
    prev_action: chex.Array
    path_cumlen: chex.Array
    path_progress: chex.Scalar


class RoboTaxiObservation(equinox.Module):
    collision_rays: chex.Array
    distance_to_path: chex.Array
    direction_of_path: chex.Array
    ray_distances: chex.Array
    features: chex.Array

    def as_vector(self) -> chex.Array:
        return jnp.concatenate([self.ray_distances, self.features], axis=-1)


def obs_size(num_ray_sensors: int) -> int:
    return num_ray_sensors + 4 + 3 + 2 * len(LOOKAHEAD) + 1


def plan_static_path(map_str, start_rc, goal_rc, path_length: int) -> np.ndarray:
    walls = np.array([[c == "#" for c in row] for row in map_str])
    h, w = walls.shape
    wall_cells = np.argwhere(walls)
    cells = np.indices((h, w)).reshape(2, -1).T
    clearance = np.min(
        np.linalg.norm(cells[:, None, :] - wall_cells[None], axis=-1), axis=1
    ).reshape(h, w)
    cell_cost = 1.0 + 4.0 * np.maximum(0.0, 2.0 - clearance)

    start, goal = tuple(int(v) for v in start_rc), tuple(int(v) for v in goal_rc)
    dist = {start: 0.0}
    parent = {}
    queue = [(0.0, start)]
    moves = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    while queue:
        d, (r, c) = heapq.heappop(queue)
        if (r, c) == goal:
            break
        if d > dist[(r, c)]:
            continue
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or walls[nr, nc]:
                continue
            if dr and dc and (walls[r + dr, c] or walls[r, c + dc]):
                continue
            nd = d + np.hypot(dr, dc) * cell_cost[nr, nc]
            if nd < dist.get((nr, nc), np.inf):
                dist[(nr, nc)] = nd
                parent[(nr, nc)] = (r, c)
                heapq.heappush(queue, (nd, (nr, nc)))
    if goal not in dist:
        raise ValueError("Goal is unreachable on this map")

    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path = path[::-1]
    if len(path) > path_length:
        raise ValueError(f"path_length={path_length} is too short ({len(path)} needed)")
    path += [goal] * (path_length - len(path))

    rc = np.array(path, dtype=np.float32)
    return np.stack([rc[:, 1], h - 1 - rc[:, 0]], axis=1)


def _to_body(vec, yaw):
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    return jnp.stack([c * vec[..., 0] + s * vec[..., 1], -s * vec[..., 0] + c * vec[..., 1]], -1)


def _path_projection(pos, path, cumlen):
    a, b = path[:-1], path[1:]
    ab = b - a
    seg_len = jnp.linalg.norm(ab, axis=1)
    t = jnp.clip(jnp.sum((pos - a) * ab, axis=1) / (seg_len**2 + 1e-8), 0.0, 1.0)
    proj = a + t[:, None] * ab
    dists = jnp.linalg.norm(pos - proj, axis=1)
    i = jnp.argmin(dists)
    seg_dir = ab[i] / (seg_len[i] + 1e-8)
    rel = pos - proj[i]
    lateral = seg_dir[0] * rel[1] - seg_dir[1] * rel[0]
    return cumlen[i] + t[i] * seg_len[i], dists[i], seg_dir, i, lateral


def _physics_substep(carry, accel_cmd, steer_cmd, dt):
    pos, yaw, vx, vy, r, steer = carry

    steer_rate = jnp.clip((steer_cmd - steer) / STEER_TAU, -STEER_RATE_MAX, STEER_RATE_MAX)
    steer = steer + steer_rate * dt

    fx = MASS * accel_cmd - C_ROLL * MASS * G * jnp.tanh(vx / 0.1) - C_DRAG * vx * jnp.abs(vx)

    vx_safe = jnp.maximum(jnp.abs(vx), V_BLEND_LOW)
    alpha_f = steer - jnp.arctan2(vy + L_F * r, vx_safe)
    alpha_r = -jnp.arctan2(vy - L_R * r, vx_safe)
    fyf = jnp.clip(C_F * alpha_f, -MU * FZ_F, MU * FZ_F)
    fyr = jnp.clip(C_R * alpha_r, -MU * FZ_R, MU * FZ_R)
    vx_dyn = vx + dt * ((fx - fyf * jnp.sin(steer)) / MASS + vy * r)
    vy_dyn = vy + dt * ((fyr + fyf * jnp.cos(steer)) / MASS - vx * r)
    r_dyn = r + dt * (L_F * fyf * jnp.cos(steer) - L_R * fyr) / I_Z

    vx_kin = vx + dt * fx / MASS
    vy_kin = vx_kin * jnp.tan(steer) * L_R / (L_F + L_R)
    r_kin = vx_kin * jnp.tan(steer) / (L_F + L_R)

    lam = jnp.clip((vx - V_BLEND_LOW) / (V_BLEND_HIGH - V_BLEND_LOW), 0.0, 1.0)
    vx = jnp.clip(lam * vx_dyn + (1 - lam) * vx_kin, V_MIN, V_MAX)
    vy = lam * vy_dyn + (1 - lam) * vy_kin
    r = lam * r_dyn + (1 - lam) * r_kin

    yaw = yaw + dt * r
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    pos = pos + dt * jnp.array([vx * c - vy * s, vx * s + vy * c])
    return pos, yaw, vx, vy, r, steer


class RoboTaxiEnv(BaseEnv):

    def init_params(
        key: chex.PRNGKey,
        map_id: int,
        max_steps: int,
        path_length: int,
        discretization_scale: int = 1,
        perception_radius: float = 5,
        num_ray_sensors: int = NUM_RAY_SENSORS,
        fov: float = FOV,
        fps: float = FPS,
    ) -> Tuple[RoboTaxiParams, RoboTaxiState]:
        task = BaseTaskSampler(map_id)
        hw = (task.height, task.width)
        env_params = RoboTaxiParams(
            max_steps_in_episode=max_steps,
            fps=fps,
            step_size=1 / fps,
            map_height_width=hw,
            nav_grid_shape_dtype=(
                (task.height * discretization_scale, task.width * discretization_scale),
                jnp.float32,
            ),
            path_array_shape_dtype=((path_length, 2), jnp.float32),
            discretization_scale=discretization_scale,
            perception_radius=perception_radius,
            agent_FOV=fov,
            num_ray_sensors=num_ray_sensors,
            physics_substeps=max(1, round(PHYSICS_HZ / fps)),
        )

        to_world = jax.vmap(partial(utils.convert_to_world_view, map_shape=hw))
        path = jnp.asarray(
            plan_static_path(task.map_str, task.agent_pos, task.goal_pos, path_length)
        )
        cumlen = jnp.concatenate(
            [jnp.zeros(1), jnp.cumsum(jnp.linalg.norm(jnp.diff(path, axis=0), axis=1))]
        )
        agent_pos = utils.convert_to_world_view(task.agent_pos, hw).astype(jnp.float32)
        forward = task.agent_forward_dir.astype(jnp.float32)

        init_state = RoboTaxiState(
            time=jnp.asarray(0),
            goal_pos=utils.convert_to_world_view(task.goal_pos, hw).astype(jnp.float32),
            agent_pos=agent_pos,
            agent_forward_dir=forward,
            static_obstacles=to_world(task.static_obstacles).astype(jnp.float32),
            kinematic_obstacles=to_world(task.kinematic_obstacles).astype(jnp.float32),
            kinematic_obst_velocities=task.kinematic_obst_velocities.astype(jnp.float32),
            path_array=path,
            yaw=jnp.arctan2(forward[1], forward[0]),
            velocity=jnp.zeros(2),
            yaw_rate=jnp.asarray(0.0),
            steer=jnp.asarray(0.0),
            prev_action=jnp.zeros(2),
            path_cumlen=cumlen,
            path_progress=_path_projection(agent_pos, path, cumlen)[0],
        )
        return env_params, init_state

    @partial(jax.jit, static_argnames=("env_params",))
    def reset(
        key: chex.PRNGKey, env_params: RoboTaxiParams, init_state: RoboTaxiState | None
    ) -> Tuple[RoboTaxiObservation, RoboTaxiState]:
        if init_state is None:
            raise Exception("init_state is None is Not implemented!")
        yaw = init_state.yaw + jax.random.uniform(
            key, minval=-YAW_RESET_NOISE, maxval=YAW_RESET_NOISE
        )
        state = equinox.tree_at(
            lambda s: (s.yaw, s.agent_forward_dir),
            init_state,
            (yaw, jnp.array([jnp.cos(yaw), jnp.sin(yaw)])),
        )
        return RoboTaxiEnv.get_observation(state, env_params), state

    @partial(jax.jit, static_argnames=("env_params",))
    def step(
        key: chex.PRNGKey,
        env_state: RoboTaxiState,
        action: chex.Array,
        env_params: RoboTaxiParams,
    ) -> Tuple[RoboTaxiObservation, RoboTaxiState, chex.Scalar, chex.Array, Dict[Any, Any]]:
        action = jnp.clip(jnp.asarray(action, jnp.float32).reshape(2), -1.0, 1.0)
        accel_cmd, steer_cmd = action[0] * A_MAX, action[1] * DELTA_MAX

        dt = env_params.step_size / env_params.physics_substeps
        carry = (
            env_state.agent_pos,
            env_state.yaw,
            env_state.velocity[0],
            env_state.velocity[1],
            env_state.yaw_rate,
            env_state.steer,
        )
        carry = jax.lax.fori_loop(
            0,
            env_params.physics_substeps,
            lambda _, c: _physics_substep(c, accel_cmd, steer_cmd, dt),
            carry,
        )
        pos, yaw, vx, vy, r, steer = carry
        yaw = jnp.arctan2(jnp.sin(yaw), jnp.cos(yaw))
        kinematic_obstacles = BaseEnv._move_kinematic_obstacles(env_state, env_params)

        new_state = RoboTaxiState(
            time=env_state.time + 1,
            goal_pos=env_state.goal_pos,
            agent_pos=pos,
            agent_forward_dir=jnp.array([jnp.cos(yaw), jnp.sin(yaw)]),
            static_obstacles=env_state.static_obstacles,
            kinematic_obstacles=kinematic_obstacles,
            kinematic_obst_velocities=env_state.kinematic_obst_velocities,
            path_array=env_state.path_array,
            yaw=yaw,
            velocity=jnp.array([vx, vy]),
            yaw_rate=r,
            steer=steer,
            prev_action=action,
            path_cumlen=env_state.path_cumlen,
            path_progress=_path_projection(pos, env_state.path_array, env_state.path_cumlen)[0],
        )

        obstacles = jnp.concatenate([env_state.static_obstacles, kinematic_obstacles], axis=0)
        h, w = env_params.map_height_width
        out_of_map = (pos[0] < -0.5) | (pos[0] > w - 0.5) | (pos[1] < -0.5) | (pos[1] > h - 0.5)
        crashed = BaseEnv._check_collisions(pos, obstacles) | out_of_map
        reached = BaseEnv._check_goal(pos, env_state.goal_pos) & ~crashed
        timeout = new_state.time >= env_params.max_steps_in_episode
        done = reached | crashed | timeout

        obs = RoboTaxiEnv.get_observation(new_state, env_params)

        progress = new_state.path_progress - env_state.path_progress
        reward = (
            W_PROGRESS * progress
            - W_PATH * jnp.minimum(obs.distance_to_path**2, 4.0)
            - W_TIME
            - W_SMOOTH * jnp.sum((action - env_state.prev_action) ** 2)
            + R_GOAL * reached
            - R_CRASH * crashed
        )

        info = {
            "time": new_state.time,
            "distance_to_path": obs.distance_to_path,
            "direction_of_path": obs.direction_of_path,
            "speed": vx,
            "reached": reached,
            "crashed": crashed,
        }
        return obs, new_state, reward, done, info

    def get_observation(
        env_state: RoboTaxiState, env_params: RoboTaxiParams
    ) -> RoboTaxiObservation:
        yaw = env_state.yaw
        path = env_state.path_array
        _, dist, seg_dir, idx, lateral = _path_projection(
            env_state.agent_pos, path, env_state.path_cumlen
        )

        obstacles = jnp.concatenate(
            [env_state.static_obstacles, env_state.kinematic_obstacles], axis=0
        )
        rays, ray_lengths = BaseEnv._collision_ray_intersections(
            env_state.agent_pos, env_state.agent_forward_dir, obstacles, env_params
        )

        n = path.shape[0]
        lookahead = jnp.stack(
            [path[jnp.minimum(idx + k, n - 1)] for k in LOOKAHEAD]
        ) - env_state.agent_pos
        goal_rel = env_state.goal_pos - env_state.agent_pos

        features = jnp.concatenate(
            [
                env_state.velocity / V_MAX,
                jnp.array([env_state.yaw_rate / 2.0, env_state.steer / DELTA_MAX]),
                _to_body(goal_rel, yaw) / 10.0,
                jnp.linalg.norm(goal_rel, keepdims=True) / 10.0,
                (_to_body(lookahead, yaw) / 5.0).reshape(-1),
                jnp.array([lateral]),
            ]
        )
        return RoboTaxiObservation(
            collision_rays=rays * ray_lengths[:, None],
            distance_to_path=dist,
            direction_of_path=seg_dir,
            ray_distances=ray_lengths / env_params.perception_radius,
            features=features,
        )
