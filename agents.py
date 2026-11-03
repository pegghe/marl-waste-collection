"""Agent behavior for the shared waste collection simulation."""

import mujoco
import numpy as np
from gymnasium import spaces

from raycast import Raycaster


class RobotAgent:
    """Robot observations, waiting behavior, and navigation reward."""

    material_count = 3
    ray_categories = ("plastic", "cardboard", "material_c", "obstacle", "robot", "none")
    ray_count = 9
    sorting_observation_size = 7 + 2 * material_count + ray_count * (1 + len(ray_categories))

    def __init__(self, env, name="robot"):
        self.env = env
        self.name = name
        prefix = "" if name == "robot" else f"{name}_"
        self.body_id = env.model.body(name).id
        self.visual_body_id = env.model.body(prefix + "robot_visual").id
        self.sensor_id = env.model.body(prefix + "robot_head_sensor").id
        joints = [env.model.joint(prefix + axis).id for axis in ("x", "y")]
        self.qpos_indices = env.model.jnt_qposadr[joints].copy()
        self.qvel_indices = env.model.jnt_dofadr[joints].copy()
        self.actuator_ids = np.array([
            env.model.actuator(prefix + "move_" + axis).id for axis in ("x", "y")
        ])
        self.max_turn_speed = np.deg2rad(120.0)
        self.turn_response_time = 0.3
        self.success_radius = 0.25
        self.observation_space = env.observation_space
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.ray_length = 2.0
        # Ignore this robot (group 1 or 2) and the floor (group 3).
        geomgroup = np.ones(6, dtype=np.uint8)
        geomgroup[1 if name == "robot" else 2] = 0
        geomgroup[3] = 0
        self.raycaster = Raycaster(
            env.model, env.data, {self.body_id: self.sensor_id}, geomgroup=geomgroup,
        )
        # Three horizontal angles at each elevation: see robot bodies as well
        # as low waste objects, which a single horizontal fan would miss.
        directions = np.array([
            [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), -np.sin(pitch)]
            for pitch in np.deg2rad([0, 15, 30])
            for yaw in np.deg2rad([-60, 0, 60])
        ])
        self.raycaster.setup_raycast(self.body_id, self.ray_count, 120,
                                    local_directions=directions)
        self.ray_distances = np.full(self.ray_count, -1.0)
        self.ray_geomids = np.full(self.ray_count, -1, dtype=np.int32)
        self.ray_types = np.full(self.ray_count, len(self.ray_categories) - 1, dtype=np.int32)

    @property
    def position(self):
        return self.env.data.qpos[self.qpos_indices]

    @property
    def velocity(self):
        return self.env.data.qvel[self.qvel_indices]

    def reset(self, qpos=None):
        env = self.env
        env.model.body_quat[self.visual_body_id] = [1, 0, 0, 0]
        self.go_delay = env.np_random.integers(0, 101)
        self.go = 0 if env.drone_active else int(self.go_delay == 0)
        self.completed = False
        self.specialization = int(env.np_random.integers(2)) if env.sorting else 0
        if env.sorting and env.two_robots:
            self.specialization = int(self.name == "robot_2")
        self.detected_type = -1
        if qpos is None:
            qpos = env.init_qpos.copy()
        position = env.np_random.uniform(-env.spawn_half_size, env.spawn_half_size, size=2)
        if self.name == "robot":
            env.target = env.np_random.uniform(-env.spawn_half_size, env.spawn_half_size, size=2)
            while np.linalg.norm(env.target - position) < env.min_spawn_distance:
                env.target = env.np_random.uniform(-env.spawn_half_size, env.spawn_half_size, size=2)
        else:
            # Keep the second shell clear of the first robot and the bottle.
            while (np.linalg.norm(position - qpos[env.robot.qpos_indices]) < 1.0
                   or np.linalg.norm(position - env.target) < env.min_spawn_distance):
                position = env.np_random.uniform(-env.spawn_half_size, env.spawn_half_size, size=2)
        qpos[self.qpos_indices] = position
        self.start_position = position.copy()
        return qpos

    def control(self, action):
        if self.env.sorting and self.env.two_robots and self.name != self.env.active_robot:
            return np.zeros(2)
        # As in the thesis, ground-agent actions stay zero until detection.
        if self.env.drone_active and not self.env.shared_comm[0]:
            return np.zeros(2)
        if self.completed:
            return np.zeros(2)
        return np.clip(action, -1, 1)

    def apply_action(self, action):
        """Advance one shared step, leaving other robot controls at zero."""
        controls = np.zeros(self.env.model.nu)
        controls[self.actuator_ids] = action
        self.env.advance_robots(controls)

    def face_target(self):
        env = self.env
        if not self.go:
            return
        if env.drone_active:
            if not env.shared_comm[0]:
                return
            target = env.shared_comm[1:3] * env.drone.area_half_size
        else:
            target = env.target
        direction = target - self.position
        if np.linalg.norm(direction) < 1e-8:
            return
        quat = env.model.body_quat[self.visual_body_id]
        yaw = 2 * np.arctan2(quat[3], quat[0])
        target_yaw = np.arctan2(direction[1], direction[0])
        # Follow the shortest arc, easing into alignment without an initial snap.
        error = (target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
        turn = error * (-np.expm1(-env.dt / self.turn_response_time))
        max_turn = self.max_turn_speed * env.dt
        half_yaw = 0.5 * (yaw + np.clip(turn, -max_turn, max_turn))
        # The decorative shell faces +X; keep the XY joints and collision body unchanged.
        env.model.body_quat[self.visual_body_id] = [
            np.cos(half_yaw), 0, 0, np.sin(half_yaw),
        ]
        mujoco.mj_forward(env.model, env.data)

    def get_observation(self):
        # The robot only receives the target broadcast after drone detection.
        if self.env.drone_active:
            if not (self.env.sorting and self.env.two_robots):
                self.go = int(bool(self.env.shared_comm[0]))
            relative_target = (
                self.env.shared_comm[1:3] * self.env.drone.area_half_size - self.position
                if self.env.shared_comm[0] else np.zeros(2)
            )
        else:
            relative_target = self.env.target - self.position
        navigation = np.concatenate((
            self.position,
            self.velocity,
            relative_target,
            [self.go],
        ))
        if not self.env.sorting:
            return navigation
        if self.env.shared_comm[3]:
            self.detected_type = int(self.env.shared_comm[3]) - 1
        return np.concatenate((
            navigation,
            [self.specialization == kind for kind in range(self.material_count)],
            [self.detected_type == kind for kind in range(self.material_count)],
            self.get_ray_observation().ravel(),
        )).astype(np.float32)

    def compute_reward(self):
        distance = float(np.linalg.norm(self.env.target - self.position))
        if self.completed:
            return 0.0, True, distance
        success = bool(self.go and distance <= self.success_radius)
        reward = -0.01  # Step penalty.
        if self.go:
            reward -= distance
        else:
            reward -= np.linalg.norm(self.velocity)
            reward -= 5 * np.linalg.norm(self.position - self.start_position)
        if self.env.sorting:
            compatible = self.specialization == self.env.target_type
            identified = self.detected_type >= 0
            if not compatible and distance <= self.success_radius:
                # Entering the collection radius of the wrong waste is a failure.
                return float(reward - 10), False, distance
            if self.env.two_robots and self.name != self.env.active_robot:
                return 0.0, False, distance
            success = bool(self.go and identified and (not compatible or distance <= self.success_radius))
            if success and not compatible:
                # Material: 0 unknown, 1 plastic, 2 cardboard, 3 reserved.
                self.env.shared_comm[:] = [1, *self.env.target / 5, self.detected_type + 1]
        if success:
            reward += 10
            self.completed = True

        # The next action receives go=1 once the waiting period is over.
        if not self.env.drone_active and not (self.env.sorting and self.env.two_robots):
            self.go = int(self.env.steps >= self.go_delay)
        return float(reward), success, distance

    def update_rays(self):
        geomids, distances = self.raycaster.perform_raycast(self.body_id, self.ray_length)
        self.ray_geomids[:] = geomids
        self.ray_distances[:] = distances
        self.ray_types.fill(len(self.ray_categories) - 1)
        hits = geomids >= 0
        self.ray_types[hits] = self.env.ray_geom_types[geomids[hits]]
        seen = self.ray_types[self.ray_types < self.material_count]
        if seen.size:
            self.detected_type = int(seen[0])

    def get_ray_observation(self):
        """Nine rows: normalized distance and one-hot semantic categories."""
        observation = np.zeros((self.ray_count, 1 + len(self.ray_categories)), dtype=np.float32)
        observation[:, 0] = np.where(
            self.ray_distances < 0, self.ray_length, self.ray_distances,
        ) / self.ray_length
        observation[np.arange(self.ray_count), 1 + self.ray_types] = 1
        return observation

    def render_rays(self, viewer):
        color = (1.0, 0.8, 0.05, 1.0) if self.name == "robot" else (0.1, 0.75, 1.0, 1.0)
        self.raycaster.render_rays(self.body_id, self.ray_length, viewer, color)


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
        self.min_search_speed = 0.3
        self.boundary_penalty = 0.2
        self.stagnation_penalty = 0.05
        self.blocked_fraction = 0.0
        self.last_movement = 0.0
        self.debug_actions = False
        target_bodies = ("target", "paper") if env.sorting else ("target",)
        target_body_ids = [env.model.body(name).id for name in target_bodies]
        self.target_geom_ids = np.flatnonzero(np.isin(env.model.geom_bodyid, target_body_ids))
        self.cell_size = 0.5
        # Rows follow world Y and columns follow world X, starting at the lower bounds.
        map_size = int(np.ceil(2 * env.area_half_size / self.cell_size))
        self.visited_map = np.zeros((map_size, map_size), dtype=np.float32)
        self.discovery_bonus = 20.0
        # Keep the existing coordinate scale for observations and target communication.
        self.area_half_size = 5.0
        self.observation_space = spaces.Box(
            low=np.array([0, -1, -1, -1, -1] + [0] * (18 + self.visited_map.size),
                         dtype=np.float32),
            high=1.0, dtype=np.float32,
        )

    def cell(self):
        """Return the current (row, column), keeping the outer boundaries in the map."""
        position = self.env.data.xpos[self.body_id, :2]
        xy = np.floor((position + self.env.area_half_size) / self.cell_size).astype(int)
        xy = np.clip(xy, 0, self.visited_map.shape[0] - 1)
        return int(xy[1]), int(xy[0])

    def get_observation(self):
        ranges = np.where(self.ray_distances < 0, self.ray_length,
                          self.ray_distances) / self.ray_length
        rays = np.zeros((9, 2), dtype=np.float32)
        rays[:, 0] = ranges
        rays[:, 1] = self.is_bottle
        position = self.env.data.xpos[self.body_id, :2] / self.area_half_size
        return np.concatenate((
            self.env.shared_comm[:3], np.clip(position, -1, 1), rays.ravel(),
            self.visited_map.ravel(),
        )).astype(np.float32)

    def compute_reward(self):
        components = {
            "time": 0.0, "exploration": 0.0, "discovery": 0.0,
            "boundary": 0.0, "stagnation": 0.0,
        }
        cell = self.cell()
        new_cell = not self.visited_map[cell]
        self.visited_map[cell] = 1.0
        if not self.env.shared_comm[0]:
            detected = self.is_bottle.any()
            if new_cell:
                components["exploration"] = 0.5
            if detected:
                remaining = max(0.0, 1.0 - (self.env.steps - 1) / self.env.max_steps)
                components["discovery"] = self.discovery_bonus * remaining
                self.env.shared_comm[0] = 1.0
                self.env.shared_comm[1:3] = self.env.target / self.area_half_size
            else:
                components["time"] = -0.05
                components["boundary"] = -self.boundary_penalty * self.blocked_fraction
                slow_fraction = np.clip(
                    1 - self.last_movement / (self.min_search_speed * self.env.dt),
                    0, 1,
                )
                components["stagnation"] = -self.stagnation_penalty * slow_fraction
        self.reward_components = components
        self.reward = sum(components.values())

    def update_rays(self):
        for i, sensor_id in enumerate(self.sensor_ids):
            geomids, distances = self.raycaster.perform_raycast(sensor_id, self.ray_length)
            self.ray_geomids[i] = geomids[0]
            self.ray_distances[i] = distances[0]
        self.is_bottle = np.isin(self.ray_geomids, self.target_geom_ids)

    def apply_action(self, action):
        position = self.env.data.mocap_pos[self.mocap_id] #actual position of the drone
        step_length = self.speed * self.env.dt #max step_length (1*0.05)= 0.05 meters
        requested = position[:2] + np.clip(action, -1, 1) * step_length #add the spostamento to the actual position
        #the drone must stay inside the boards limits (-3.5, +3.5)
        destination = np.clip(
            requested,
            -self.env.area_half_size, self.env.area_half_size,
        )
        # Measure actual progress, including the part of the command rejected at an edge.
        self.blocked_fraction = float(np.clip(
            np.linalg.norm(requested - destination) / step_length, 0, 1,
        ))
        self.last_movement = float(np.linalg.norm(destination - position[:2]))
        position[:2] = destination
        

    def reset_position(self, randomize=False):
        if randomize:
            self.env.data.mocap_pos[self.mocap_id, :2] = self.env.np_random.uniform(
                -self.env.spawn_half_size, self.env.spawn_half_size, size=2,
            )

    def reset(self):
        self.update_rays()
        self.visited_map.fill(0.0)
        self.visited_map[self.cell()] = 1.0
        self.blocked_fraction = 0.0
        self.last_movement = 0.0
        self.reward = 0.0
        self.reward_components = {
            "time": 0.0, "exploration": 0.0, "discovery": 0.0,
            "boundary": 0.0, "stagnation": 0.0,
        }

    def render_rays(self, viewer):
        self.update_rays()
        for sensor_id in self.sensor_ids:
            self.raycaster.render_rays(
                sensor_id, self.ray_length, viewer, (1.0, 0.05, 0.02, 1.0),
            )
