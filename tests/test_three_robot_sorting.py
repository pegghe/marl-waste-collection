"""Three specialists identify, hand off and collect all three waste categories."""

from contextlib import redirect_stdout
from io import StringIO
from itertools import combinations, permutations
import unittest
from unittest.mock import Mock, patch

import numpy as np
from masb3.utils import ma_evaluate

from waste_collection_env import RobotMAEnv, WasteCollectionEnv


class ThreeRobotSortingTest(unittest.TestCase):
    def setUp(self):
        self.ma_env = RobotMAEnv(sorting=True, three_robots=True)
        self.addCleanup(self.ma_env.close)
        self.ma_env.reset(seed=42)
        self.env = self.ma_env.env
        self.actions = {name: np.zeros(2) for name in self.env.robots}

    def place(self, material, nearest):
        env = self.env
        env.target_type = material
        env.target = np.array([1.2, 0.0])
        env.shared_comm.fill(0)
        env.team_reward_given = False
        for kind, body in enumerate(env.material_bodies):
            env.model.body(body).pos[:2] = env.target if kind == material else [20, 20]
        qpos = env.data.qpos.copy()
        for role, robot in enumerate(env.robots.values()):
            robot.completed = False
            robot.detected_type = -1
            env.model.body_quat[robot.visual_body_id] = [1, 0, 0, 0]
            qpos[robot.qpos_indices] = [0, 0] if role == nearest else [-3, role * 2 - 2]
        env.set_state(qpos, np.zeros(env.model.nv))
        env.assign_robot()

    def test_reset_assigns_three_roles_and_samples_only_one_of_three_objects(self):
        materials = set()
        for seed in range(32):
            observations, info = self.ma_env.reset(seed=seed)
            positions = {name: robot.position.copy() for name, robot in self.env.robots.items()}
            materials.add(self.env.target_type)
            self.assertEqual(info['robot_specializations'], {
                'robot': 'plastic', 'robot_2': 'cardboard', 'robot_3': 'residual',
            })
            self.assertEqual(self.env.model.nu, 6)
            for role, (name, robot) in enumerate(self.env.robots.items()):
                self.assertGreaterEqual(np.linalg.norm(robot.position - self.env.target), 3)
                self.assertEqual(observations[name].shape, (76,))
                self.assertTrue(robot.observation_space.contains(observations[name]))
                np.testing.assert_array_equal(observations[name][7:10], np.eye(3)[role])
            for first, second in combinations(positions.values(), 2):
                self.assertGreaterEqual(np.linalg.norm(first - second), 1)
            for kind, body in enumerate(self.env.material_bodies):
                expected = self.env.target if kind == self.env.target_type else [20, 20]
                np.testing.assert_array_equal(self.env.model.body(body).pos[:2], expected)
            repeated, _ = self.ma_env.reset(seed=seed)
            for name, robot in self.env.robots.items():
                np.testing.assert_array_equal(robot.position, positions[name])
                np.testing.assert_array_equal(observations[name], repeated[name])
        self.assertEqual(materials, {0, 1, 2})

    def test_masb3_predicts_three_actions_with_one_shared_policy(self):
        model = Mock()
        model.predict.side_effect = lambda obs: (np.zeros((len(obs), 2)), None)
        self.ma_env.set_agent_models({'robot': model})
        for agent in self.ma_env.agents.values():
            self.assertIs(agent.model, model)
        self.env.max_steps = 2
        with redirect_stdout(StringIO()):
            per_agent, per_model = ma_evaluate(self.ma_env, {'robot': model}, total_episodes=1)
        self.assertEqual(model.predict.call_count, 2)
        for call in model.predict.call_args_list:
            self.assertEqual(call.args[0].shape, (3, 76))
        self.assertAlmostEqual(per_model['robot'], sum(per_agent.values()) / 3)

    def test_every_incompatible_reporter_hands_off_to_the_correct_specialist(self):
        for material, reporter_role in permutations(range(3), 2):
            with self.subTest(material=material, reporter=reporter_role):
                self.place(material, nearest=reporter_role)
                robots = list(self.env.robots.values())
                collector, reporter = robots[material], robots[reporter_role]
                observations, _, terminated, truncated, info = self.ma_env.step_all(self.actions)
                self.assertFalse(terminated or truncated)
                self.assertTrue(info['reported'])
                self.assertFalse(info['collected'])
                self.assertEqual(info['team_reward'], 0)
                self.assertTrue(reporter.completed)
                self.assertEqual(info['active_robot'], collector.name)
                np.testing.assert_allclose(info['shared_comm'], [1, 1.2 / 5, 0, material + 1])
                for name, robot in self.env.robots.items():
                    self.assertEqual(observations[name][6], int(name == collector.name))
                    self.assertEqual(observations[name][10 + material], 1)
                self.assertFalse(np.any(collector.ray_types == material))
                qpos = self.env.data.qpos.copy()
                qpos[reporter.qpos_indices] = [-3, reporter_role * 2 - 2]
                qpos[collector.qpos_indices] = [1.0, 0]
                self.env.set_state(qpos, np.zeros(self.env.model.nv))
                _, rewards, terminated, truncated, info = self.ma_env.step_all(self.actions)
                self.assertTrue(terminated)
                self.assertFalse(truncated)
                self.assertTrue(info['is_success'])
                self.assertTrue(info['collected'])
                self.assertFalse(info['wrong_target'])
                self.assertIsNone(info['active_robot'])
                self.assertEqual(info['team_reward'], 10)
                for name, robot in self.env.robots.items():
                    self.assertEqual(rewards[name], info['individual_rewards'][name] + 10)
                    if robot is not collector:
                        self.assertEqual(rewards[name], 10)
                _, _, _, _, repeated = self.ma_env.step_all(self.actions)
                self.assertEqual(repeated['team_reward'], 0)

    def test_each_specialist_can_collect_without_reporting(self):
        for material in range(3):
            with self.subTest(material=material):
                self.place(material, nearest=material)
                _, _, terminated, _, info = self.ma_env.step_all(self.actions)
                self.assertFalse(terminated)
                self.assertFalse(info['reported'])
                collector = list(self.env.robots.values())[material]
                self.assertEqual(collector.detected_type, material)
                qpos = self.env.data.qpos.copy()
                qpos[collector.qpos_indices] = [1.0, 0]
                self.env.set_state(qpos, np.zeros(self.env.model.nv))
                _, _, terminated, _, info = self.ma_env.step_all(self.actions)
                self.assertTrue(terminated)
                self.assertTrue(info['is_success'])
                self.assertTrue(info['collected'])
                self.assertFalse(info['reported'])

    def test_wrong_collection_fails_for_every_incompatible_specialist(self):
        for material, wrong_role in permutations(range(3), 2):
            with self.subTest(material=material, wrong=wrong_role):
                self.place(material, nearest=wrong_role)
                wrong = list(self.env.robots.values())[wrong_role]
                qpos = self.env.data.qpos.copy()
                qpos[wrong.qpos_indices] = [1.0, 0]
                self.env.set_state(qpos, np.zeros(self.env.model.nv))
                _, rewards, terminated, _, info = self.ma_env.step_all(self.actions)
                self.assertTrue(terminated)
                self.assertTrue(info['wrong_target'])
                self.assertFalse(info['is_success'])
                self.assertEqual(info['team_reward'], 0)
                self.assertLess(rewards[wrong.name], -10)

    def test_waiting_robots_stay_still_and_physics_advances_once(self):
        for active in range(3):
            with self.subTest(active=active):
                self.place(active, nearest=active)
                idle = [r for r in self.env.robots.values() if r.name != self.env.active_robot]
                positions = [r.position.copy() for r in idle]
                for robot in idle:
                    self.env.data.qvel[robot.qvel_indices] = [1, -1]
                start_time = self.env.data.time
                with patch.object(self.env, 'do_simulation', wraps=self.env.do_simulation) as simulate:
                    _, _, _, _, info = self.ma_env.step_all({name: [1, 1] for name in self.actions})
                self.assertEqual(simulate.call_count, 1)
                self.assertAlmostEqual(self.env.data.time - start_time, self.env.dt)
                for robot, position in zip(idle, positions):
                    np.testing.assert_array_equal(robot.position, position)
                    np.testing.assert_array_equal(robot.velocity, 0)
                    np.testing.assert_array_equal(self.env.data.ctrl[robot.actuator_ids], 0)
                    self.assertEqual(info['individual_rewards'][robot.name], 0)

    def test_every_head_sees_both_teammates_and_ignores_its_own_shell(self):
        robots = list(self.env.robots.values())
        self.env.model.body('target').pos[:2] = [-20, -20]
        self.env.model.body('paper').pos[:2] = [-20, -20]
        self.env.model.body('material_c').pos[:2] = [-20, -20]
        for own, other in permutations(robots, 2):
            with self.subTest(own=own.name, other=other.name):
                qpos = self.env.data.qpos.copy()
                for role, robot in enumerate(robots):
                    qpos[robot.qpos_indices] = [-3, role * 2 - 2]
                qpos[own.qpos_indices] = [0, 0]
                qpos[other.qpos_indices] = [1, 0]
                self.env.set_state(qpos, np.zeros(self.env.model.nv))
                own.update_rays()
                self.assertEqual(own.ray_types[1], 4)
                self.assertEqual(self.env.model.geom_group[own.ray_geomids[1]], other.geom_group)
                hits = own.ray_geomids[own.ray_geomids >= 0]
                self.assertFalse(np.any(self.env.model.geom_group[hits] == own.geom_group))

    def test_avoidance_detects_proximity_and_contact_with_either_teammate(self):
        robots = list(self.env.robots.values())
        for own, other in permutations(robots, 2):
            with self.subTest(own=own.name, other=other.name):
                own.go = 1
                def place(distance):
                    qpos = self.env.data.qpos.copy()
                    for role, robot in enumerate(robots):
                        qpos[robot.qpos_indices] = [-3, role * 2 - 2]
                    qpos[own.qpos_indices] = [0, 0]
                    qpos[other.qpos_indices] = [distance, 0]
                    self.env.set_state(qpos, np.zeros(self.env.model.nv))
                    return own.compute_avoidance_reward()
                self.assertEqual(place(1.2), 0)
                self.assertGreater(place(0.6), -own.robot_collision_penalty)
                self.assertLess(place(0.6), 0)
                self.assertLessEqual(place(0.3), -own.robot_collision_penalty)
                own.go = 0
                self.assertEqual(own.compute_avoidance_reward(), 0)

    def test_drone_finds_each_material_before_dispatch_and_stops_after_discovery(self):
        env = RobotMAEnv(sorting=True, three_robots=True, drone_search=True)
        self.addCleanup(env.close)
        for material in range(3):
            with self.subTest(material=material):
                env.reset(seed=42)
                world = env.env
                world.target_type = material
                world.target = np.array([1.2, 0])
                for kind, body in enumerate(world.material_bodies):
                    world.model.body(body).pos[:2] = world.target if kind == material else [20, 20]
                qpos = world.data.qpos.copy()
                for robot, position in zip(world.robots.values(), ([-2, -2], [-2, 2], [-3, 0])):
                    qpos[robot.qpos_indices] = position
                world.data.mocap_pos[world.drone.mocap_id] = [-3, 3, 3]
                world.set_state(qpos, np.zeros(world.model.nv))
                world.assign_robot()
                positions = {name: robot.position.copy() for name, robot in world.robots.items()}
                actions = {name: [1, 1] for name in world.robots}
                actions['drone'] = [0, 0]
                observations, _, _, _, info = env.step_all(actions)
                self.assertFalse(info['drone_found_target'])
                self.assertIsNone(info['active_robot'])
                for name, robot in world.robots.items():
                    np.testing.assert_array_equal(robot.position, positions[name])
                    np.testing.assert_array_equal(observations[name][4:7], 0)
                world.data.mocap_pos[world.drone.mocap_id, :2] = world.target
                observations, _, terminated, _, info = env.step_all(actions)
                self.assertFalse(terminated)
                self.assertTrue(info['drone_found_target'])
                self.assertEqual(info['active_robot'], 'robot')
                np.testing.assert_allclose(info['shared_comm'], [1, 1.2 / 5, 0, 0])
                self.assertEqual(observations['drone'].shape, (219,))
                self.assertTrue(world.drone_observation_space.contains(observations['drone']))
                position = world.data.mocap_pos[world.drone.mocap_id].copy()
                world.move_drone([1, 1])
                np.testing.assert_array_equal(world.data.mocap_pos[world.drone.mocap_id], position)

    def test_three_robots_require_sorting_and_exclude_two_robot_flag(self):
        for kwargs in ({'three_robots': True}, {'sorting': True, 'two_robots': True, 'three_robots': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                WasteCollectionEnv(**kwargs)


if __name__ == '__main__':
    unittest.main()
