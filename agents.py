"""Agent behavior for the shared waste collection simulation."""

import mujoco
import numpy as np
from gymnasium import spaces

from raycast import Raycaster


class RobotAgent:
    """Robot observations, waiting behavior, and navigation reward."""

    def __init__(self, env):
        self.env = env
        self.success_radius = 0.25
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self):
        env = self.env
        self.go_delay = env.np_random.integers(0, 101)
        self.go = int(self.go_delay == 0)
        qpos = env.init_qpos.copy()
        qpos[:2] = env.np_random.uniform(-4, 4, size=2)
        self.start_position = qpos[:2].copy()
        env.target = env.np_random.uniform(-4, 4, size=2)
        while np.linalg.norm(env.target - qpos[:2]) < 0.75:
            env.target = env.np_random.uniform(-4, 4, size=2)
        return qpos

    def apply_action(self, action):
        self.env.do_simulation(np.clip(action, -1, 1), self.env.frame_skip)

    def get_observation(self):
        # Position, velocity, relative target coordinates, go signal.
        return np.concatenate((
            self.env.data.qpos[:2],
            self.env.data.qvel[:2],
            self.env.target - self.env.data.qpos[:2],
            [self.go],
        ))

    def compute_reward(self):
        distance = float(np.linalg.norm(self.env.target - self.env.data.qpos[:2]))
        success = bool(self.go and distance <= self.success_radius)
        reward = -0.01  # Step penalty.
        if self.go:
            reward -= distance
        else:
            reward -= np.linalg.norm(self.env.data.qvel[:2])
            reward -= 5 * np.linalg.norm(self.env.data.qpos[:2] - self.start_position)
        if success:
            reward += 10

        # The next action receives go=1 once the waiting period is over.
        self.go = int(self.env.steps >= self.go_delay)
        return float(reward), success, distance


class DroneAgent:
    """Drone movement, ray sensors, observations, and exploration reward."""

    def __init__(self, env):
        self.env = env
        self.sensor_ids = [self.env.model.body(f"drone_sensor_{i}").id for i in range(9)]
        self.ray_length = 5.0
        self.raycaster = Raycaster(
            self.env.model, self.env.data, {sensor_id: sensor_id for sensor_id in self.sensor_ids},
        )
        for sensor_id in self.sensor_ids:
            self.raycaster.setup_raycast(sensor_id, nrays=1, angle_covered_degrees=0)
        self.ray_distances = np.full(9, -1.0)
        self.ray_geomids = np.full(9, -1, dtype=np.int32)
        self.body_id = self.env.model.body("drone").id
        self.mocap_id = int(self.env.model.body_mocapid[self.body_id])
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.speed = 1.0
        target_body_id = self.env.model.body("target").id
        self.target_geom_ids = np.flatnonzero(self.env.model.geom_bodyid == target_body_id)
        self.cell_size = 0.5
        self.discovery_bonus = 20.0
        self.area_half_size = 5.0
        self.observation_space = spaces.Box(
            low=np.array([0, -1, -1, -1, -1] + [0] * 18, dtype=np.float32),
            high=1.0, dtype=np.float32,
        )

    def cell(self):
        position = self.env.data.xpos[self.body_id, :2]
        return tuple(np.floor(position / self.cell_size).astype(int))

    def get_observation(self):
        ranges = np.where(self.ray_distances < 0, self.ray_length,
                          self.ray_distances) / self.ray_length
        rays = np.zeros((9, 2), dtype=np.float32)
        rays[:, 0] = ranges
        rays[:, 1] = self.is_bottle
        position = self.env.data.xpos[self.body_id, :2] / self.area_half_size
        return np.concatenate((
            self.env.shared_comm, np.clip(position, -1, 1), rays.ravel(),
        )).astype(np.float32)

    def compute_reward(self):
        components = {"time": 0.0, "exploration": 0.0, "discovery": 0.0}
        if not self.env.shared_comm[0]:
            detected = self.is_bottle.any()
            cell = self.cell()
            if cell not in self.visited_cells:
                self.visited_cells.add(cell)
                components["exploration"] = 0.5
            if detected:
                remaining = max(0.0, 1.0 - (self.env.steps - 1) / self.env.max_steps)
                components["discovery"] = self.discovery_bonus * remaining
                self.env.shared_comm[0] = 1.0
                self.env.shared_comm[1:] = self.env.target / self.area_half_size
            else:
                components["time"] = -0.05
        self.reward_components = components
        self.reward = sum(components.values())

    def update_rays(self):
        for i, sensor_id in enumerate(self.sensor_ids):
            geomids, distances = self.raycaster.perform_raycast(sensor_id, self.ray_length)
            self.ray_geomids[i] = geomids[0]
            self.ray_distances[i] = distances[0]
        self.is_bottle = np.isin(self.ray_geomids, self.target_geom_ids)

    def apply_action(self, action):
        position = self.env.data.mocap_pos[self.mocap_id]
        position[:2] = np.clip(
            position[:2] + np.clip(action, -1, 1) * self.speed * self.env.dt, -5, 5,
        )

    def reset_position(self, randomize=False):
        if randomize:
            self.env.data.mocap_pos[self.mocap_id, :2] = self.env.np_random.uniform(-4, 4, size=2)

    def reset(self):
        self.update_rays()
        self.visited_cells = {self.cell()}
        self.reward = 0.0
        self.reward_components = {"time": 0.0, "exploration": 0.0, "discovery": 0.0}

    def render_rays(self, viewer):
        self.update_rays()
        for sensor_id, distance in zip(self.sensor_ids, self.ray_distances):
            length = self.ray_length if distance < 0 else distance
            origin = self.raycaster.raycast_origins_buf[sensor_id]
            direction = self.raycaster.raycast_ray_dirs_buf[sensor_id][0]
            end = origin + direction * length
            geom = mujoco.MjvGeom()
            mujoco.mjv_initGeom(
                geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3),
                np.eye(3).ravel(), np.array([1.0, 0.05, 0.02, 1.0], dtype=np.float32),
            )
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2.0, origin, end)
            viewer.add_marker(
                type=mujoco.mjtGeom.mjGEOM_LINE,
                pos=geom.pos.copy(), mat=geom.mat.ravel().copy(),
                size=geom.size.copy(), rgba=geom.rgba.copy(), label="",
            )
