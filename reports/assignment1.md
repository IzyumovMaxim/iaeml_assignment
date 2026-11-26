# Assignment 1: Environment

Maxim Iziumov, AAI-02

The environment is implemented in [src/env/robotaxi.py](../src/env/robotaxi.py).

## 1. Dynamics model

### State and action spaces

The action is a vector $a \in [-1, 1]^2$. The first component is the longitudinal acceleration command, $a_x = 3 a_0$ m/s². When the car moves forward, a negative value works as a brake. When the car stands still, it drives the car in reverse. The second component is the steering command, $\delta_{cmd} = 0.5 a_1$ rad, which is about 29 degrees.

The vehicle state consists of the position $(X, Y)$, the heading $\psi$, the body-frame velocities $v_x$ and $v_y$, the yaw rate $r$ and the actual steering angle $\delta$. The environment state also keeps everything from BaseEnvState (goal, static and moving obstacles, path), the previous action, the cumulative arc length of the path and the progress of the agent along the path.

One map cell is 1 m. The agent is a small robotaxi with mass $m = 150$ kg, distances from the centre of gravity to the axles $l_f = l_r = 0.45$ m and yaw inertia $I_z = 20$ kg·m². Its speed is limited to the range from -1.5 to 4 m/s.

### Equations

I use a dynamic bicycle model with a linear tyre model and friction saturation.

Slip angles of the front and rear axles use $v_{\mathrm{safe}} = \max(|v_x|, 0.5)$, with velocities measured in m/s:

$$
\alpha_f = \delta - \arctan\frac{v_y + l_f r}{v_{\mathrm{safe}}}
$$

$$
\alpha_r = -\arctan\frac{v_y - l_r r}{v_{\mathrm{safe}}}
$$

Lateral tyre forces are linear in the slip angle and limited by friction:

$$
F_{y,i} = \mathrm{clip}\left(C_i \alpha_i,\ -\mu F_{z,i},\ \mu F_{z,i}\right), \quad i \in \{f, r\}
$$

$$
F_{z,f} = m g \frac{l_r}{l_f + l_r}, \qquad F_{z,r} = m g \frac{l_f}{l_f + l_r}
$$

Here $C_f = C_r = 2500$ N/rad and $\mu = 0.9$.

The longitudinal force is traction or braking minus rolling resistance and aerodynamic drag:

$$
F_x = m a_x - c_{rr} m g \tanh(v_x / 0.1) - c_d v_x \lvert v_x \rvert
$$

Here $c_{rr} = 0.02$ and $c_d = 3$. The $\tanh$ is a smooth replacement for the sign function, so the car does not jitter around zero speed.

Dynamics in the body frame:

$$
\dot v_x = \frac{F_x - F_{y,f} \sin\delta}{m} + v_y r
$$

$$
\dot v_y = \frac{F_{y,r} + F_{y,f} \cos\delta}{m} - v_x r
$$

$$
\dot r = \frac{l_f F_{y,f} \cos\delta - l_r F_{y,r}}{I_z}
$$

$$
\dot\psi = r, \qquad \dot X = v_x \cos\psi - v_y \sin\psi, \qquad \dot Y = v_x \sin\psi + v_y \cos\psi
$$

The steering actuator is a first-order lag with a rate limit, so the wheels cannot turn instantly:

$$
\dot\delta = \mathrm{clip}\left(\frac{\delta_{cmd} - \delta}{0.1},\ -2,\ 2\right) \ \text{rad/s}
$$

Near standstill the slip-angle calculation is sensitive to small changes in $v_x$. The lower bound on $v_{\mathrm{safe}}$ keeps the calculation well defined, and at low speed the dynamic model is smoothly blended with the kinematic bicycle model, where $v_y = v_x \tan\delta \cdot l_r / L$ and $r = v_x \tan\delta / L$:

$$
\lambda = \mathrm{clip}\left(\frac{v_x - 0.5}{1.5 - 0.5},\ 0,\ 1\right), \qquad x = \lambda x_{dyn} + (1 - \lambda) x_{kin}
$$

Below 0.5 m/s, including reverse, the model is purely kinematic. Above 1.5 m/s it is purely dynamic.

The equations are integrated with the semi-implicit Euler method at 120 Hz. One environment step of 1/fps seconds is split into $\mathrm{round}(120 / \mathrm{fps})$ substeps inside jax.lax.fori_loop, so the control frequency and the physics frequency are independent.

### Justification

The kinematic model cannot describe skidding or understeer, and the point model in base.py does not even have the non-holonomic constraint. The dynamic bicycle model is the simplest model that has both. It still needs only a few arithmetic operations per step, so it vectorises well in JAX.

Tyre saturation at $\mu F_z$ gives a physical limit on lateral acceleration. At high speed with full steering the car turns wider than the geometry alone would suggest.

## 2. Observation space

The observation is a vector of length $M + 12$, which is 28 for $M = 16$ rays. The goal and lookahead vectors are expressed in the vehicle's body frame.

It contains the $M$ ray lengths (rays cover 360 degrees, perception radius 5 m) divided by 5. Then four vehicle values: $v_x / V_{max}$, $v_y / V_{max}$, $r / 2$ and $\delta / \delta_{max}$. Then the vector to the goal divided by 10 and the distance to the goal divided by 10, three values in total. Then two path points ahead of the agent, 3 and 6 waypoints after the nearest one, divided by 5, four values in total. The last value is the signed lateral deviation from the path.

The body frame makes the policy invariant to rotation and translation of the map. The rays in the base code are already cast relative to the heading of the agent.

The global path is built once in init_params with Dijkstra search on the 8-connected grid of static walls. The cost of a cell grows near walls as $1 + 4 \max(0,\ 2 - d_{wall})$, and diagonal moves through a wall corner are not allowed. The planner is plain Python and NumPy. It is the only part of the environment that is not written in JAX, and it is not called during a step. I wrote my own planner because, at the default grid resolution, `BaseEnv._find_path` can route close to walls and cut corners diagonally. Moving obstacles are not part of the planner. The agent sees them through the rays, so the global planner and the local reactive avoidance are separated.

## 3. Reward function

$$
R_t = 1.0 \cdot \Delta s_t - 0.02 \min(d_t^2, 4) - 0.01 - 0.01 \lVert a_t - a_{t-1} \rVert^2 + 10 \cdot [\text{goal}] - 10 \cdot [\text{crash}]
$$

The first term $(1.0 \cdot \Delta s_t)$ is the increase of the arc length of the projection of the agent onto the path. It gives the agent a signal at every step based on how far it moves along the route. Over the whole path of about 22 m the agent collects about 22, which is the same order as the goal bonus.

The second term $(-0.02 \min(d_t^2, 4))$ penalises deviation from the path. It is small and capped, because the agent must be allowed to leave the path to go around a moving obstacle.

The third term $(-0.01)$ is a time penalty that makes the agent drive instead of standing still. Standing for 450 steps costs -4.5, which is less in absolute value than a crash (-10). So standing still to avoid a crash is worse than reaching the goal (+10 +22).

The fourth term $(-0.01 \lVert a_t - a_{t-1} \rVert^2)$ penalises changes of the action and removes steering chatter.

The fifth term $(+10 \cdot [\text{goal}])$ adds 10 reward points on the step when the agent reaches the goal. Here $[\text{goal}]$ is 1 on that step and 0 otherwise.

The last term $(-10 \cdot [\text{crash}])$ subtracts 10 reward points on the step when a crash occurs. Here $[\text{crash}]$ is 1 on that step and 0 otherwise. A crash is a collision with a static or moving obstacle (the distance threshold from BaseEnv._check_collisions) or leaving the map. Map 2 has a gap in the wall, so leaving the map is possible there.

An episode ends when the goal is reached, after a crash, or after max_steps. Unlike BaseEnv.step, the check is done after the movement, so the reward and done refer to the same state.

## 4. Difficulties and reflection

The default planner sometimes cut corners next to walls. I changed the planner to block those diagonal moves and assign a higher cost to cells close to walls. To see how much this mattered in practice, I ran both routes on map 0 with the same vehicle dynamics, starting state, and pure pursuit controller. With the default route, the car hit a wall after 40 steps. With the new route, it reached the goal after 174 steps.

When I compared my model with the initial version that used only the dynamic model, I found a couple of bugs:

1. If the car stands still and you press A and D one after another quickly, the car turns in place, which real cars do not do
2. If you press S+A or S+D, the car turns in the wrong direction, as if the steering were inverted in reverse

The first bug turns out to happen because near standstill the slip-angle equations are sensitive to small changes in longitudinal speed. The second one happens because the slip angle is computed with $|v_x|$, so the formula always assumes that the car moves forward. The kinematic model fixes both: the yaw rate is $r = v_x \tan\delta / L$, so it is zero when the car stands still and changes sign in reverse. I used the kinematic bicycle model in reverse and at forward speeds up to 0.5 m/s. Between 0.5 and 1.5 m/s, I gradually increased the weight of the dynamic model. From 1.5 m/s onward, the update uses the dynamic model alone.

## 5. Results and reproduction

To demonstrate the dynamics I use a classic pure pursuit controller that follows the path and ignores obstacles. It is implemented in [src/utils/robotaxi_frontend.py](../src/utils/robotaxi_frontend.py).

On map 0 it reaches the goal with return about 29 in 174 steps (11.6 s): [media/pursuit_map0.mp4](../media/pursuit_map0.mp4).

On map 1 it crashes into the moving obstacle, because the controller does not see it: [media/pursuit_map1.mp4](../media/pursuit_map1.mp4). This is the motivation for RL in Assignment 2.

On map 2, pure pursuit also collides with a moving obstacle: [media/pursuit_map2.mp4](../media/pursuit_map2.mp4).

To reproduce on macOS or Linux with Python 3.11 or newer:

```bash
pip install -r requirements.txt
# manual driving: W/S throttle and brake, A/D steering
JAX_PLATFORM_NAME=cpu python src/play.py
# pure pursuit in a window
JAX_PLATFORM_NAME=cpu python src/play.py --controller pursuit --map-id 1
# record a video without a window (mp4 needs ffmpeg, a .gif path works without it)
JAX_PLATFORM_NAME=cpu python src/play.py --controller pursuit --map-id 0 --record media/pursuit_map0.mp4 --episodes 2
```
