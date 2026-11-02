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
                 two_robots=False, sorting=False):
        if sorting and (two_robots or drone_training):
            raise ValueError("Sorting training uses one robot without a drone policy")
        self.sorting = sorting
        self.two_robots = two_robots
        self.drone_training = drone_training
        self.finish_on_discovery = drone_training if finish_on_discovery is None else finish_on_discovery
        super().__init__(
            model_path=str(Path(__file__).with_name("scene.xml")),
            frame_skip=5,
            observation_space=spaces.Box(
                -np.inf, np.inf, shape=(65 if sorting else 7,),
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
        self.min_spawn_distance = 3.0 if sorting else 0.75
        self.max_steps = 2000
        self.robot = RobotAgent(self)
        self.robots = {"robot": self.robot}
        if two_robots:
            self.robots["robot_2"] = RobotAgent(self, "robot_2")
        self.drone = DroneAgent(self)
        # Ray categories: plastic, paper, obstacle, other robot, no hit.
        self.ray_geom_types = np.full(self.model.ngeom, 2, dtype=np.int32)
        self.ray_geom_types[np.isin(self.model.geom_group, [1, 2])] = 3
        self.ray_geom_types[self.drone.target_geom_ids] = 0
        if sorting:
            self.ray_geom_types[self.model.geom_bodyid == self.model.body("paper").id] = 1
        self.drone_action_space = self.drone.action_space
        self.drone_observation_space = self.drone.observation_space

    def _initialize_simulation(self):
        """Build the second robot from the same shell, joints and sensor definition."""
        scene = ET.parse(self.fullpath).getroot()
        world = scene.find("worldbody")
        if not self.sorting:
            world.remove(world.find("body[@name='paper']"))
        robot = world.find("body[@name='robot']")
        for geom in robot.iter("geom"):
            geom.set("group", "1")
        world.find("geom[@name='floor']").set("group", "3")
        if self.two_robots:
            second = deepcopy(robot)
            for element in second.iter():
                if "name" in element.attrib:
                    name = element.get("name")
                    element.set("name", "robot_2" if name == "robot" else "robot_2_" + name)
                if element.tag == "geom":
                    element.set("group", "2")
                    if element.get("rgba") == "0.96 0.70 0.04 1":
                        element.set("rgba", "0.10 0.55 0.95 1")
            world.append(second)
            actuators = scene.find("actuator")
            for motor in list(actuators):
                duplicate = deepcopy(motor)
                duplicate.set("name", "robot_2_" + motor.get("name"))
                duplicate.set("joint", "robot_2_" + motor.get("joint"))
                actuators.append(duplicate)
        model = mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding="unicode"))
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
        # Group 3 is hidden by default in MuJoCo; here it contains the floor.
        viewer.vopt.geomgroup[3] = 1
        self.drone.render_rays(viewer)
        for robot in self.robots.values():
            robot.update_rays()
            robot.render_rays(viewer)
        return super().render()

    def _get_obs(self):
        return self.robot.get_observation()

    def reset_model(self):
        self.steps = 0
        qpos = self.robot.reset()
        if self.two_robots:
            qpos = self.robots["robot_2"].reset(qpos)
        self.target_type = int(self.np_random.integers(2)) if self.sorting else 0
        if self.sorting:
            for kind, body in enumerate(("target", "paper")):
                self.model.body(body).pos[:2] = self.target if kind == self.target_type else [20, 20]
        else:
            self.model.body("target").pos[:2] = self.target
        self.drone.reset_position(randomize=self.drone_training)
        self.set_state(qpos, self.init_qvel.copy())
        self.shared_comm = np.zeros(4 if self.sorting else 3, dtype=np.float32)
        self.drone.reset()
        for robot in self.robots.values():
            robot.update_rays()
        return self._get_obs()

    def move_drone(self, action):
        self.drone.apply_action(action)

    def advance_robots(self, action):
        """Apply every motor command in one shared physics step."""
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.model.nu,):
            raise ValueError(f"Expected joint robot action shape {(self.model.nu,)}")
        controls = np.zeros(self.model.nu)
        for robot in self.robots.values():
            controls[robot.actuator_ids] = robot.control(action[robot.actuator_ids])
        self.do_simulation(controls, self.frame_skip)
        for robot in self.robots.values():
            robot.face_target()

    def step(self, action):
        self.advance_robots(action)
        self.steps += 1
        for robot in self.robots.values():
            robot.update_rays()
        results = {name: robot.compute_reward() for name, robot in self.robots.items()}
        reward, _, distance = results["robot"]
        success = all(result[1] for result in results.values())
        self.drone.update_rays()
        if not self.sorting:
            self.drone.compute_reward()
        if self.render_mode == "human":
            self.render()

        info = {
            "distance": distance, "is_success": success,
            "robot_rewards": {name: result[0] for name, result in results.items()},
            "robot_successes": {name: result[1] for name, result in results.items()},
            "robot_sensors": self.get_robot_sensors(),
            "drone_ray_distances": self.drone.ray_distances.copy(),
            "drone_ray_geomids": self.drone.ray_geomids.copy(),
            "drone_reward": self.drone.reward,
            "drone_reward_components": self.drone.reward_components.copy(),
            "drone_found_target": bool(self.shared_comm[0]) and not self.sorting,
            "drone_observation": self.get_drone_observation(),
            "drone_visited_map": self.drone.visited_map.copy(),
            "drone_speed": self.drone.last_movement / self.dt,
            "drone_blocked_fraction": self.drone.blocked_fraction,
        }
        terminated = bool(self.shared_comm[0]) if self.finish_on_discovery else success
        info["is_success"] = terminated
        if self.sorting:
            wrong_target = (
                self.robot.specialization != self.target_type and distance <= self.robot.success_radius
            )
            terminated = success or wrong_target
            info.update(
                is_success=success, wrong_target=wrong_target,
                collected=success and self.robot.specialization == self.target_type,
                reported=bool(self.shared_comm[0]),
                shared_comm=self.shared_comm.copy(),
                specialization=RobotAgent.ray_categories[self.robot.specialization],
                target_type=RobotAgent.ray_categories[self.target_type],
            )
        truncated = self.steps >= self.max_steps and not terminated
        return self._get_obs(), float(reward), terminated, truncated, info


class RobotMAEnv(BaseMAEnv):
    def __init__(self, render_mode=None, drone_training=False, finish_on_discovery=None,
                 two_robots=False, sorting=False):
        super().__init__()
        self.drone_training = drone_training
        self.episode_successes = []
        self.env = WasteCollectionEnv(
            render_mode=render_mode, drone_training=drone_training,
            finish_on_discovery=finish_on_discovery,
            two_robots=two_robots,
            sorting=sorting,
        )
        for name, robot in self.env.robots.items():
            self.register_agent(name, robot.observation_space, robot.action_space,
                                model_name="robot")
        if drone_training:
            self.register_agent("drone", self.env.drone_observation_space, self.env.drone_action_space)

    def reset(self, seed=None):
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
        if self.drone_training:
            self.env.move_drone(self.drone_action)
        self.observation, self.reward, self.terminated, self.truncated, self.info = (
            self.env.step(self.action)
        )
        if self.terminated or self.truncated:
            self.episode_successes.append(bool(self.info["is_success"]))

    def get_observation(self, agent_id):
        if agent_id == "drone":
            return self.env.get_drone_observation()
        return self.env.robots[agent_id].get_observation()

    def evaluate_env_state(self):
        rewards = self.info["robot_rewards"].copy()
        if self.drone_training:
            rewards["drone"] = self.env.drone_reward
        return rewards, self.terminated, self.truncated, self.info

    def close(self):
        self.env.close()
