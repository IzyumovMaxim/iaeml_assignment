import json
from dataclasses import asdict, dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from env.robotaxi import YAW_RESET_NOISE, RoboTaxiEnv, obs_size


@dataclass(frozen=True)
class PPOConfig:
    map_id: int = 0
    fps: int = 15
    max_steps: int = 450
    num_ray_sensors: int = 16
    path_length: int = 100
    perception_radius: float = 5.0

    num_envs: int = 64
    num_steps: int = 128
    total_timesteps: int = 5_000_000
    lr: float = 3e-4
    anneal_lr: bool = True
    update_epochs: int = 4
    num_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 128
    seed: int = 0
    start_frac: float = 0.0
    timeout_penalty: float = 0.0

    @property
    def num_updates(self) -> int:
        return self.total_timesteps // (self.num_envs * self.num_steps)


class ActorCritic(eqx.Module):
    actor: eqx.nn.MLP
    critic: eqx.nn.MLP
    log_std: jax.Array

    def __init__(self, obs_dim: int, act_dim: int, hidden: int, key):
        k1, k2 = jax.random.split(key)
        actor = eqx.nn.MLP(obs_dim, act_dim, hidden, 2, activation=jnp.tanh, key=k1)
        last = actor.layers[-1]
        self.actor = eqx.tree_at(lambda m: m.layers[-1].weight, actor, last.weight * 0.01)
        self.critic = eqx.nn.MLP(obs_dim, "scalar", hidden, 2, activation=jnp.tanh, key=k2)
        self.log_std = jnp.full((act_dim,), -0.5)

    def __call__(self, obs):
        return self.actor(obs), self.log_std, self.critic(obs)


def make_env(cfg: PPOConfig):
    return RoboTaxiEnv.init_params(
        jax.random.PRNGKey(cfg.seed),
        map_id=cfg.map_id,
        max_steps=cfg.max_steps,
        path_length=cfg.path_length,
        perception_radius=cfg.perception_radius,
        num_ray_sensors=cfg.num_ray_sensors,
        fps=cfg.fps,
    )


def make_model(cfg: PPOConfig, key):
    return ActorCritic(obs_size(cfg.num_ray_sensors), 2, cfg.hidden, key)


def save_checkpoint(path: str, model: ActorCritic, cfg: PPOConfig):
    with open(path, "wb") as f:
        f.write((json.dumps(asdict(cfg)) + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load_checkpoint(path: str):
    with open(path, "rb") as f:
        cfg = PPOConfig(**json.loads(f.readline().decode()))
        model = eqx.tree_deserialise_leaves(f, make_model(cfg, jax.random.PRNGKey(0)))
    return model, cfg


def gaussian_log_prob(x, mean, log_std):
    return jnp.sum(
        -0.5 * ((x - mean) / jnp.exp(log_std)) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi),
        axis=-1,
    )


def make_train(cfg: PPOConfig):
    env_params, init_state = make_env(cfg)
    batch_size = cfg.num_envs * cfg.num_steps
    minibatch_size = batch_size // cfg.num_minibatches

    path, cumlen = init_state.path_array, init_state.path_cumlen
    goal_idx = jnp.argmax(cumlen >= cumlen[-1])

    def start_on_path(key, state):
        k_idx, k_yaw = jax.random.split(key)
        i = jax.random.randint(k_idx, (), 0, goal_idx - 1)
        d = path[i + 1] - path[i]
        yaw = jnp.arctan2(d[1], d[0]) + jax.random.uniform(
            k_yaw, minval=-YAW_RESET_NOISE, maxval=YAW_RESET_NOISE
        )
        return eqx.tree_at(
            lambda s: (s.agent_pos, s.yaw, s.agent_forward_dir, s.path_progress),
            state,
            (path[i], yaw, jnp.array([jnp.cos(yaw), jnp.sin(yaw)]), cumlen[i]),
        )

    def env_reset(key):
        k_reset, k_coin, k_path = jax.random.split(key, 3)
        _, state = RoboTaxiEnv.reset(k_reset, env_params, init_state)
        on_path = jax.random.uniform(k_coin) < cfg.start_frac
        state = jax.tree.map(
            lambda a, b: jnp.where(on_path, a, b), start_on_path(k_path, state), state
        )
        return RoboTaxiEnv.get_observation(state, env_params).as_vector(), state

    def env_step(key, state, action):
        k_step, k_reset = jax.random.split(key)
        obs, next_state, reward, done, info = RoboTaxiEnv.step(k_step, state, action, env_params)
        timeout = done & ~info["reached"] & ~info["crashed"]
        reward = reward - cfg.timeout_penalty * timeout
        reset_obs, reset_state = env_reset(k_reset)
        next_state = jax.tree.map(lambda r, s: jnp.where(done, r, s), reset_state, next_state)
        obs = jnp.where(done, reset_obs, obs.as_vector())
        return obs, next_state, reward, done, info

    v_reset = jax.vmap(env_reset)
    v_step = jax.vmap(env_step)

    def lr_schedule(count):
        frac = 1.0 - (count // (cfg.num_minibatches * cfg.update_epochs)) / cfg.num_updates
        return cfg.lr * frac

    tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(lr_schedule if cfg.anneal_lr else cfg.lr, eps=1e-5),
    )

    def init(key):
        k_model, k_env, key = jax.random.split(key, 3)
        model = make_model(cfg, k_model)
        params, static = eqx.partition(model, eqx.is_array)
        obs, env_state = v_reset(jax.random.split(k_env, cfg.num_envs))
        zeros = jnp.zeros(cfg.num_envs)
        runner = (params, tx.init(params), env_state, obs, zeros, zeros, key)
        return runner, static

    def update(runner, static):
        params, opt_state, env_state, obs, ep_ret, ep_len, key = runner
        model = eqx.combine(params, static)

        def rollout_step(carry, _):
            env_state, obs, ep_ret, ep_len, key = carry
            key, k_act, k_env = jax.random.split(key, 3)
            mean, log_std, value = jax.vmap(model)(obs)
            action = mean + jnp.exp(log_std) * jax.random.normal(k_act, mean.shape)
            log_prob = gaussian_log_prob(action, mean, log_std)
            next_obs, env_state, reward, done, info = v_step(
                jax.random.split(k_env, cfg.num_envs), env_state, action
            )
            ep_ret, ep_len = ep_ret + reward, ep_len + 1
            transition = dict(
                obs=obs, action=action, log_prob=log_prob, value=value,
                reward=reward, done=done,
                ep_return=ep_ret, ep_length=ep_len,
                reached=info["reached"], crashed=info["crashed"],
            )
            ep_ret, ep_len = ep_ret * (1 - done), ep_len * (1 - done)
            return (env_state, next_obs, ep_ret, ep_len, key), transition

        (env_state, obs, ep_ret, ep_len, key), traj = jax.lax.scan(
            rollout_step, (env_state, obs, ep_ret, ep_len, key), None, cfg.num_steps
        )

        _, _, last_value = jax.vmap(model)(obs)

        def gae_step(carry, t):
            gae, next_value = carry
            not_done = 1.0 - t["done"]
            delta = t["reward"] + cfg.gamma * next_value * not_done - t["value"]
            gae = delta + cfg.gamma * cfg.gae_lambda * not_done * gae
            return (gae, t["value"]), gae

        _, advantages = jax.lax.scan(
            gae_step, (jnp.zeros_like(last_value), last_value), traj, reverse=True
        )
        targets = advantages + traj["value"]

        def loss_fn(params, batch):
            mean, log_std, value = jax.vmap(eqx.combine(params, static))(batch["obs"])
            log_prob = gaussian_log_prob(batch["action"], mean, log_std)

            value_clipped = batch["value"] + jnp.clip(
                value - batch["value"], -cfg.clip_eps, cfg.clip_eps
            )
            value_loss = 0.5 * jnp.maximum(
                (value - batch["target"]) ** 2, (value_clipped - batch["target"]) ** 2
            ).mean()

            adv = batch["adv"]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ratio = jnp.exp(log_prob - batch["log_prob"])
            actor_loss = -jnp.minimum(
                ratio * adv, jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv
            ).mean()
            entropy = jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e))

            total = actor_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy
            approx_kl = ((ratio - 1) - jnp.log(ratio)).mean()
            return total, (actor_loss, value_loss, entropy, approx_kl)

        data = dict(
            obs=traj["obs"], action=traj["action"], log_prob=traj["log_prob"],
            value=traj["value"], adv=advantages, target=targets,
        )
        data = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), data)

        def epoch(carry, _):
            params, opt_state, key = carry
            key, k_perm = jax.random.split(key)
            perm = jax.random.permutation(k_perm, batch_size)
            shuffled = jax.tree.map(
                lambda x: x[perm].reshape((cfg.num_minibatches, minibatch_size) + x.shape[1:]),
                data,
            )

            def minibatch(carry, batch):
                params, opt_state = carry
                (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
                updates, opt_state = tx.update(grads, opt_state, params)
                return (optax.apply_updates(params, updates), opt_state), aux

            (params, opt_state), aux = jax.lax.scan(minibatch, (params, opt_state), shuffled)
            return (params, opt_state, key), aux

        (params, opt_state, key), aux = jax.lax.scan(
            epoch, (params, opt_state, key), None, cfg.update_epochs
        )

        done = traj["done"]
        n_done = jnp.maximum(done.sum(), 1)
        metrics = dict(
            episodes=done.sum(),
            ep_return=(traj["ep_return"] * done).sum() / n_done,
            ep_length=(traj["ep_length"] * done).sum() / n_done,
            success=(traj["reached"] * done).sum() / n_done,
            crash=(traj["crashed"] * done).sum() / n_done,
            actor_loss=aux[0].mean(),
            value_loss=aux[1].mean(),
            entropy=aux[2].mean(),
            approx_kl=aux[3].mean(),
        )
        runner = (params, opt_state, env_state, obs, ep_ret, ep_len, key)
        return runner, metrics

    return init, eqx.filter_jit(update)
