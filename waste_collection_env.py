"""Shared MuJoCo simulation and MASB3 coordination for robot and drone."""

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from masb3.base import BaseMAEnv

from agents import DroneAgent, RobotAgent


class WasteCollectionEnv(MujocoEnv):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 20,
    }

    def __init__(self, render_mode=None, drone_training=False, finish_on_discovery=None,
                 two_robots=False, sorting=False, drone_search=False, three_robots=False,
                 robots_per_team=None, waste_per_type=None):
        if sorting and drone_training:
            raise ValueError("Sorting mode does not use a drone policy")
        if two_robots and three_robots:
            raise ValueError("Choose either two or three robots")
        if three_robots and not sorting:
            raise ValueError("Three robots require sorting mode")
        self.collection_mode = robots_per_team is not None or waste_per_type is not None
        if self.collection_mode and not sorting:
            raise ValueError("Configurable teams and waste counts require sorting mode")
        if robots_per_team is not None and (two_robots or three_robots):
            raise ValueError("Use robots_per_team instead of the fixed robot flags")
        if self.collection_mode and drone_search:
            raise ValueError("Configurable collection currently uses known positions; omit the drone model")
        self.sorting = sorting
        self.two_robots = two_robots
        self.three_robots = three_robots
        self.robots_per_team = None
        self.waste_per_type = None
        if self.collection_mode:
            self.robots_per_team = tuple(robots_per_team) if robots_per_team is not None else (
                (1, 1, 0) if two_robots else (1, 1, 1)
            )
            self.waste_per_type = tuple(waste_per_type) if waste_per_type is not None else tuple(
                int(count > 0) for count in self.robots_per_team
            )
            for label, counts in (("robots_per_team", self.robots_per_team),
                                  ("waste_per_type", self.waste_per_type)):
                if len(counts) != 3 or any(count < 0 for count in counts) or sum(counts) == 0:
                    raise ValueError(f"{label} needs three nonnegative counts with a positive total")
            if any(waste and not robots for robots, waste in zip(self.robots_per_team, self.waste_per_type)):
                raise ValueError("Every waste category needs at least one compatible robot")
        self.robot_count = sum(self.robots_per_team) if self.collection_mode else (
            3 if three_robots else (2 if two_robots else 1)
        )
        self.multi_robot = self.robot_count > 1
        self.dispatching = sorting and (self.multi_robot or self.collection_mode)
        self.robot_names = tuple("robot" if index == 0 else f"robot_{index + 1}"
                                 for index in range(self.robot_count))
        self.robot_roles = dict(zip(self.robot_names, (
            np.repeat(np.arange(3), self.robots_per_team) if self.collection_mode
            else np.arange(self.robot_count)
        )))
        if self.collection_mode:
            self.material_types = np.repeat(np.arange(3), self.waste_per_type)
            self.material_bodies = tuple(f"waste_{index}" for index in range(len(self.material_types)))
        else:
            self.material_bodies = ("target", "paper", "material_c")[:
                3 if three_robots else (2 if sorting else 1)
            ]
            self.material_types = np.arange(len(self.material_bodies))
        self.drone_training = drone_training
        self.drone_active = drone_training or drone_search
        self.finish_on_discovery = drone_training if finish_on_discovery is None else finish_on_discovery
        super().__init__(
            model_path=str(Path(__file__).with_name("scene.xml")),
            frame_skip=5,
            observation_space=spaces.Box(
                -np.inf, np.inf, shape=(RobotAgent.sorting_observation_size if sorting else 7,),
                dtype=np.float32 if sorting else np.float64,
            ),
            render_mode=render_mode,
            default_camera_config={
                "distance": 12.0, "elevation": -35.0, "azimuth": 135.0,
                "lookat": np.array([0.0, 0.0, 1.0]),
            },
        )
        self.area_half_size = 3.5
        self.spawn_half_size = 3.0
        self.min_spawn_distance = 3.0 if sorting and not self.collection_mode else 0.75
        self.max_steps = 2000
        self.team_bonus = 10.0
        self.robot = RobotAgent(self)
        self.robots = {"robot": self.robot}
        for name in self.robot_names[1:]:
            self.robots[name] = RobotAgent(self, name)
        self.drone = DroneAgent(self)
        # Three material slots, then obstacle, other robot and no hit.
        self.ray_geom_types = np.full(self.model.ngeom, RobotAgent.material_count, dtype=np.int32)
        self.geom_robot_ids = np.full(self.model.ngeom, -1, dtype=np.int32)
        for index, robot in enumerate(self.robots.values()):
            self.ray_geom_types[robot.geom_ids] = RobotAgent.material_count + 1
            self.geom_robot_ids[robot.geom_ids] = index
        self.waste_body_ids = np.array([self.model.body(body).id for body in self.material_bodies])
        self.geom_waste_ids = np.full(self.model.ngeom, -1, dtype=np.int32)
        for index, (kind, body_id) in enumerate(zip(self.material_types, self.waste_body_ids)):
            geoms = self.model.geom_bodyid == body_id
            self.ray_geom_types[geoms] = kind
            self.geom_waste_ids[geoms] = index
        self.drone_action_space = self.drone.action_space
        self.drone_observation_space = self.drone.observation_space

    def _initialize_simulation(self):
        """Build each specialist from the same shell, joints and sensor definition."""
        scene = ET.parse(self.fullpath).getroot()
        world = scene.find("worldbody")
        waste_templates = [world.find(f"body[@name='{name}']")
                           for name in ("target", "paper", "material_c")]
        for template in waste_templates:
            if template.get("name") not in self.material_bodies:
                world.remove(template)
        if self.collection_mode:
            for name, kind in zip(self.material_bodies, self.material_types):
                item = deepcopy(waste_templates[kind])
                item.set("name", name)
                for element in item.iter():
                    if element is not item and "name" in element.attrib:
                        element.set("name", name + "_" + element.get("name"))
                world.append(item)
        robot = world.find("body[@name='robot']")
        colors = ("0.96 0.70 0.04 1", "0.10 0.55 0.95 1", "0.15 0.80 0.30 1")
        robot_template = deepcopy(robot)
        if self.collection_mode:
            for geom in robot.iter("geom"):
                if geom.get("rgba") == colors[0]:
                    geom.set("rgba", colors[self.robot_roles["robot"]])
        for geom in robot.iter("geom"):
            geom.set("group", "1")
        world.find("geom[@name='floor']").set("group", "3")
        actuators = scene.find("actuator")
        original_motors = list(actuators)
        for index, name in enumerate(self.robot_names[1:], start=1):
            # MuJoCo has six geometry groups, so additional robots share a group.
            group = (1, 2, 4)[index] if index < 3 else 1
            color = colors[self.robot_roles[name] if self.collection_mode else index]
            duplicate_robot = deepcopy(robot_template)
            for element in duplicate_robot.iter():
                if "name" in element.attrib:
                    original_name = element.get("name")
                    element.set("name", name if original_name == "robot" else name + "_" + original_name)
                if element.tag == "geom":
                    element.set("group", str(group))
                    if element.get("rgba") == "0.96 0.70 0.04 1":
                        element.set("rgba", color)
            world.append(duplicate_robot)
            for motor in original_motors:
                duplicate = deepcopy(motor)
                duplicate.set("name", name + "_" + motor.get("name"))
                duplicate.set("joint", name + "_" + motor.get("joint"))
                actuators.append(duplicate)
        assets = {
            mesh.get("file"): (Path(self.fullpath).parent / mesh.get("file")).read_bytes()
            for mesh in scene.findall("asset/mesh[@file]")
        }
        model = mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding="unicode"), assets=assets)
        model.vis.global_.offwidth = self.width
        model.vis.global_.offheight = self.height
        return model, mujoco.MjData(model)

    @property
    def drone_reward(self):
        return self.drone.reward

    def get_drone_observation(self):
        return self.drone.get_observation()

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        info["drone_observation"] = self.get_drone_observation()
        info["drone_visited_map"] = self.drone.visited_map.copy()
        info["robot_sensors"] = self.get_robot_sensors()
        if self.sorting:
            info["shared_comm"] = self.shared_comm.copy()
            if self.dispatching:
                info["active_robot"] = self.active_robot
            info["robot_specializations"] = {
                name: RobotAgent.ray_categories[robot.specialization]
                for name, robot in self.robots.items()
            }
        if self.collection_mode:
            info.update(self.get_collection_info())
        return obs, info

    def get_robot_sensors(self):
        """Return distances and geometry IDs for each robot."""
        return {
            name: {
                "distances": robot.ray_distances.copy(),
                "geomids": robot.ray_geomids.copy(),
                "types": robot.ray_types.copy(),
            }
            for name, robot in self.robots.items()
        }

    def render(self):
        viewer = self.mujoco_renderer._get_viewer(self.render_mode)
        # Groups 3 and 4 contain the floor and the third robot.
        viewer.vopt.geomgroup[3] = 1
        viewer.vopt.geomgroup[4] = 1
        self.drone.render_rays(viewer)
        for robot in self.robots.values():
            robot.update_rays()
        # Pose only the rendered frame, leaving sensing and physics in their base pose.
        try:
            for robot in self.robots.values():
                robot.pose_legs()
            mujoco.mj_kinematics(self.model, self.data)
            return super().render()
        finally:
            for robot in self.robots.values():
                self.model.body_quat[robot.leg_body_ids] = [1, 0, 0, 0]
            mujoco.mj_kinematics(self.model, self.data)

    def _get_obs(self):
        return self.robot.get_observation()

    def reset_model(self):
        self.steps = 0
        self.team_reward_given = False
        qpos = self.robot.reset()
        for name in self.robot_names[1:]:
            qpos = self.robots[name].reset(qpos)
        if self.collection_mode:
            self.waste_positions = np.zeros((len(self.material_bodies), 2))
            self.waste_collected = np.zeros(len(self.material_bodies), dtype=bool)
            self.known_waste_types = np.full(len(self.material_bodies), -1, dtype=np.int32)
            self.current_waste = None
            robot_positions = np.array([qpos[robot.qpos_indices] for robot in self.robots.values()])
            for index, body_id in enumerate(self.waste_body_ids):
                for _ in range(2000):
                    position = self.np_random.uniform(-self.spawn_half_size, self.spawn_half_size, size=2)
                    if (np.all(np.linalg.norm(robot_positions - position, axis=1) >= self.min_spawn_distance)
                            and np.all(np.linalg.norm(self.waste_positions[:index] - position, axis=1) >= 0.75)):
                        break
                else:
                    raise ValueError("Not enough room to spawn the requested waste; reduce waste counts")
                self.waste_positions[index] = position
                self.model.body_pos[body_id, :2] = position
        else:
            self.target_type = int(self.np_random.integers(len(self.material_bodies))) if self.sorting else 0
        if self.sorting and not self.collection_mode:
            for kind, body in enumerate(self.material_bodies):
                self.model.body(body).pos[:2] = self.target if kind == self.target_type else [20, 20]
        elif not self.sorting:
            self.model.body("target").pos[:2] = self.target
        self.drone.reset_position(randomize=self.drone_active)
        self.set_state(qpos, self.init_qvel.copy())
        self.shared_comm = np.zeros(4 if self.sorting else 3, dtype=np.float32)
        if self.collection_mode:
            self.select_next_waste()
        elif self.dispatching:
            self.assign_robot()
        self.drone.reset()
        for robot in self.robots.values():
            robot.update_rays()
        return self._get_obs()

    def select_next_waste(self):
        pending = np.flatnonzero(~self.waste_collected)
        for robot in self.robots.values():
            robot.completed = False
            robot.detected_type = -1
            robot.go = 0
            self.data.qvel[robot.qvel_indices] = 0
        self.shared_comm.fill(0)
        self.active_robot = None
        if not pending.size:
            self.current_waste = None
            return
        self.current_waste = int(min(
            pending, key=lambda index: min(
                np.linalg.norm(robot.position - self.waste_positions[index])
                for robot in self.robots.values()
            ),
        ))
        self.target = self.waste_positions[self.current_waste].copy()
        self.target_type = int(self.material_types[self.current_waste])
        known = int(self.known_waste_types[self.current_waste])
        if known >= 0:
            self.shared_comm[:] = [1, *self.target / 5, known + 1]
        self.assign_robot(material=known if known >= 0 else None)

    def get_collection_info(self):
        count = int(self.waste_collected.sum())
        return {
            "current_waste": self.current_waste,
            "waste_positions": self.waste_positions.copy(),
            "waste_materials": [RobotAgent.ray_categories[kind] for kind in self.material_types],
            "waste_collected": self.waste_collected.copy(),
            "known_waste_types": self.known_waste_types.copy(),
            "collected_count": count,
            "remaining_waste": len(self.waste_collected) - count,
            "total_waste": len(self.waste_collected),
            "collection_fraction": count / len(self.waste_collected),
        }

    def complete_collection_step(self, info):
        item = self.current_waste
        reported = bool(self.shared_comm[3]) and item is not None
        collected = item is not None and any(
            robot.completed and robot.specialization == self.target_type
            for robot in self.robots.values()
        )
        if reported:
            self.known_waste_types[item] = int(self.shared_comm[3]) - 1
        if collected:
            self.waste_collected[item] = True
            self.known_waste_types[item] = self.target_type
            self.model.body_pos[self.waste_body_ids[item], :2] = [20 + item, 20]
            mujoco.mj_forward(self.model, self.data)
            self.select_next_waste()
            for robot in self.robots.values():
                robot.update_rays()
            self.drone.update_rays()
        elif item is not None and self.robots[self.active_robot].completed:
            self.assign_robot(int(self.shared_comm[3]) - 1)
        success = bool(self.waste_collected.all())
        info.update(
            is_success=success, wrong_target=False, collected=collected, reported=reported,
            collected_waste=item if collected else None,
            reported_waste=item if reported else None,
            active_robot=self.active_robot,
            shared_comm=self.shared_comm.copy(),
            specialization=RobotAgent.ray_categories[self.robot.specialization],
            robot_specializations={
                name: RobotAgent.ray_categories[robot.specialization]
                for name, robot in self.robots.items()
            },
            target_type=RobotAgent.ray_categories[self.target_type] if self.current_waste is not None else None,
        )
        info.update(self.get_collection_info())
        # Reassignment can change the rays and go flag returned with this step.
        info["robot_sensors"] = self.get_robot_sensors()
        info["drone_ray_distances"] = self.drone.ray_distances.copy()
        info["drone_ray_geomids"] = self.drone.ray_geomids.copy()
        info["drone_observation"] = self.get_drone_observation()
        return success

    def move_drone(self, action):
        if self.sorting and self.shared_comm[0]:
            return
        self.drone.apply_action(action)

    def assign_robot(self, material=None):
        target = self.shared_comm[1:3] * self.drone.area_half_size if self.drone_active else self.target
        candidates = [
            name for name, robot in self.robots.items()
            if not robot.completed and (material is None or robot.specialization == material)
            and (not self.drone_active or self.shared_comm[0])
        ]
        self.active_robot = min(
            candidates, key=lambda name: np.linalg.norm(self.robots[name].position - target),
            default=None,
        )
        for name, robot in self.robots.items():
            robot.go = int(name == self.active_robot)
            if not robot.go:
                self.data.qvel[robot.qvel_indices] = 0

    def advance_robots(self, action):
        """Apply every motor command in one shared physics step."""
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.model.nu,):
            raise ValueError(f"Expected joint robot action shape {(self.model.nu,)}")
        controls = np.zeros(self.model.nu)
        stationary = {}
        for robot in self.robots.values():
            controls[robot.actuator_ids] = robot.control(action[robot.actuator_ids])
            if self.dispatching and robot.name != self.active_robot:
                stationary[robot.name] = robot.position.copy()
                self.data.qvel[robot.qvel_indices] = 0
        self.do_simulation(controls, self.frame_skip)
        # Keep unassigned robots stationary, including under inertia or contact.
        for name, position in stationary.items():
            robot = self.robots[name]
            self.data.qpos[robot.qpos_indices] = position
            self.data.qvel[robot.qvel_indices] = 0
        if stationary:
            mujoco.mj_forward(self.model, self.data)
        for robot in self.robots.values():
            robot.face_target()
            robot.update_gait()

    def step(self, action):
        self.advance_robots(action)
        self.steps += 1
        for robot in self.robots.values():
            robot.update_rays()
        results = {name: robot.compute_reward() for name, robot in self.robots.items()}
        distance = results["robot"][2]
        success = all(result[1] for result in results.values())
        self.drone.update_rays()
        if not self.sorting or self.drone_active:
            self.drone.compute_reward()

        info = {
            "distance": distance, "is_success": success,
            "individual_rewards": {name: result[0] for name, result in results.items()},
            "robot_avoidance_rewards": {
                name: robot.avoidance_reward for name, robot in self.robots.items()
            },
            "robot_successes": {name: result[1] for name, result in results.items()},
            "robot_sensors": self.get_robot_sensors(),
            "drone_ray_distances": self.drone.ray_distances.copy(),
            "drone_ray_geomids": self.drone.ray_geomids.copy(),
            "drone_reward": self.drone.reward,
            "drone_reward_components": self.drone.reward_components.copy(),
            "drone_found_target": bool(self.shared_comm[0]) and self.drone_active,
            "drone_observation": self.get_drone_observation(),
            "drone_visited_map": self.drone.visited_map.copy(),
            "drone_speed": self.drone.last_movement / self.dt,
            "drone_blocked_fraction": self.drone.blocked_fraction,
        }
        terminated = bool(self.shared_comm[0]) if self.finish_on_discovery else success
        info["is_success"] = terminated
        if self.collection_mode:
            success = self.complete_collection_step(info)
            terminated = success
        elif self.sorting:
            wrong_target = any(
                robot.specialization != self.target_type and results[name][2] <= robot.success_radius
                for name, robot in self.robots.items()
            )
            collected = any(
                robot.completed and robot.specialization == self.target_type
                for robot in self.robots.values()
            )
            success = (collected if self.multi_robot else success) and not wrong_target
            terminated = success or wrong_target
            if self.multi_robot:
                if terminated:
                    self.active_robot = None
                    for robot in self.robots.values():
                        robot.go = 0
                        self.data.qvel[robot.qvel_indices] = 0
                elif self.active_robot is None and self.shared_comm[0]:
                    self.assign_robot()
                elif self.active_robot is not None and self.robots[self.active_robot].completed:
                    self.assign_robot(int(self.shared_comm[3]) - 1)
                info["active_robot"] = self.active_robot
            info.update(
                is_success=success, wrong_target=wrong_target,
                collected=collected,
                reported=bool(self.shared_comm[3]),
                shared_comm=self.shared_comm.copy(),
                specialization=RobotAgent.ray_categories[self.robot.specialization],
                robot_specializations={
                    name: RobotAgent.ray_categories[robot.specialization]
                    for name, robot in self.robots.items()
                },
                target_type=RobotAgent.ray_categories[self.target_type],
            )
        truncated = self.steps >= self.max_steps and not terminated
        team_reward = 0.0
        if self.sorting and self.multi_robot and success and not self.team_reward_given:
            team_reward = self.team_bonus
            self.team_reward_given = True
        info["team_reward"] = team_reward
        info["robot_rewards"] = {
            name: individual + team_reward for name, individual in info["individual_rewards"].items()
        }
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), info["robot_rewards"]["robot"], terminated, truncated, info


class RobotMAEnv(BaseMAEnv):
    def __init__(self, render_mode=None, drone_training=False, finish_on_discovery=None,
                 two_robots=False, sorting=False, drone_search=False, three_robots=False,
                 robots_per_team=None, waste_per_type=None, seed=None):
        super().__init__()
        self.drone_training = drone_training
        self.episode_successes = []
        self.episode_collection_fractions = []
        self.initial_seed = seed
        self.env = WasteCollectionEnv(
            render_mode=render_mode, drone_training=drone_training,
            finish_on_discovery=finish_on_discovery,
            two_robots=two_robots,
            three_robots=three_robots,
            sorting=sorting,
            drone_search=drone_search,
            robots_per_team=robots_per_team,
            waste_per_type=waste_per_type,
        )
        for name, robot in self.env.robots.items():
            self.register_agent(name, robot.observation_space, robot.action_space,
                                model_name="robot")
        if self.env.drone_active:
            self.register_agent("drone", self.env.drone_observation_space, self.env.drone_action_space)

    def reset(self, seed=None):
        if seed is None:
            seed = self.initial_seed
        self.initial_seed = None
        obs, info = self.env.reset(seed=seed)
        self.observation = obs
        self.action = np.zeros(self.env.model.nu)
        self.drone_action = np.zeros(2)
        self.previous_observation = self.get_observations()
        return self.previous_observation, info

    def step_agent(self, agent_id, action):
        if agent_id == "drone":
            self.drone_action = action
        else:
            self.action[self.env.robots[agent_id].actuator_ids] = action

    def sync_wait_for_actions_completion(self):
        if self.env.drone_active:
            self.env.move_drone(self.drone_action)
        self.observation, self.reward, self.terminated, self.truncated, self.info = (
            self.env.step(self.action)
        )
        if self.terminated or self.truncated:
            self.episode_successes.append(bool(self.info["is_success"]))
            if self.env.collection_mode:
                self.episode_collection_fractions.append(self.info["collection_fraction"])

    def get_observation(self, agent_id):
        if agent_id == "drone":
            return self.env.get_drone_observation()
        return self.env.robots[agent_id].get_observation()

    def evaluate_env_state(self):
        rewards = self.info["robot_rewards"].copy()
        if self.env.drone_active:
            rewards["drone"] = self.env.drone_reward
        return rewards, self.terminated, self.truncated, self.info

    def close(self):
        self.env.close()
