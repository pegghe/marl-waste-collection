"""Configurable teams, object-specific recognition and repeated collections."""

from contextlib import redirect_stdout
from io import StringIO
from itertools import permutations
import unittest
from unittest.mock import Mock, patch

import numpy as np
from masb3.utils import ma_evaluate

from waste_collection_env import RobotMAEnv, WasteCollectionEnv


class ConfigurableCollectionTest(unittest.TestCase):
    def make_env(self, robots=(1, 1, 1), waste=(2, 2, 2), **kwargs):
        env = RobotMAEnv(sorting=True, robots_per_team=robots, waste_per_type=waste, **kwargs)
        self.addCleanup(env.close)
        env.reset(seed=42)
        return env

    def place_current(self, ma_env, nearest):
        env = ma_env.env
        # Isolate the current item from the rest of the scene for each task.
        for index, body_id in enumerate(env.waste_body_ids):
            env.waste_positions[index] = [20 + index, 20]
            env.model.body_pos[body_id, :2] = env.waste_positions[index]
        env.target = np.array([1.2, 0.0])
        env.waste_positions[env.current_waste] = env.target
        env.model.body_pos[env.waste_body_ids[env.current_waste], :2] = env.target
        env.target_type = int(env.material_types[env.current_waste])
        env.shared_comm.fill(0)
        qpos = env.data.qpos.copy()
        positions = ([-2.7, -2.5], [-2.7, 0], [-2.7, 2.5], [2.7, 2.5], [2.7, -2.5], [0, 2.7])
        for index, robot in enumerate(env.robots.values()):
            qpos[robot.qpos_indices] = [0, 0] if index == nearest else positions[index]
            env.model.body_quat[robot.visual_body_id] = [1, 0, 0, 0]
            robot.completed = False
            robot.detected_type = -1
        env.set_state(qpos, np.zeros(env.model.nv))
        env.assign_robot()
        return {name: np.zeros(2) for name in env.robots}

    def test_counts_roles_spawns_and_observations_are_repeatable(self):
        env = self.make_env(robots=(2, 2, 2), waste=(4, 4, 4))
        observations, info = env.reset(seed=7)
        positions = [r.position.copy() for r in env.env.robots.values()]
        self.assertEqual(env.env.robot_count, 6)
        self.assertEqual(env.env.model.nu, 12)
        self.assertEqual(info['total_waste'], 12)
        self.assertEqual(info['remaining_waste'], 12)
        self.assertEqual(info['collection_fraction'], 0)
        self.assertEqual(info['waste_materials'], ['plastic'] * 4 + ['cardboard'] * 4 + ['residual'] * 4)
        for index, (name, robot) in enumerate(env.env.robots.items()):
            self.assertEqual(robot.specialization, index // 2)
            self.assertEqual(observations[name].shape, (76,))
            self.assertTrue(robot.observation_space.contains(observations[name]))
            self.assertTrue(np.all(np.linalg.norm(info['waste_positions'] - robot.position, axis=1) >= 0.75))
        for first, second in permutations(positions, 2):
            self.assertGreaterEqual(np.linalg.norm(first - second), 1)
        for first, second in permutations(info['waste_positions'], 2):
            self.assertGreaterEqual(np.linalg.norm(first - second), 0.75)
        repeated, repeated_info = env.reset(seed=7)
        np.testing.assert_array_equal(info['waste_positions'], repeated_info['waste_positions'])
        for name in observations:
            np.testing.assert_array_equal(observations[name], repeated[name])

    def test_all_six_items_are_collected_once_without_resetting_the_episode(self):
        ma_env = self.make_env()
        env = ma_env.env
        snapshot = env.get_collection_info()
        seen = set()
        previous_steps = 0
        for count in range(1, 7):
            item = env.current_waste
            material = env.material_types[item]
            collector = next(r for r in env.robots.values() if r.specialization == material)
            actions = self.place_current(ma_env, collector.robot_index)
            _, _, terminated, _, _ = ma_env.step_all(actions)
            self.assertFalse(terminated)
            self.assertEqual(collector.detected_type, material)
            qpos = env.data.qpos.copy()
            qpos[collector.qpos_indices] = [1.0, 0]
            env.set_state(qpos, np.zeros(env.model.nv))
            observations, rewards, terminated, truncated, info = ma_env.step_all(actions)
            self.assertFalse(truncated)
            self.assertEqual(terminated, count == 6)
            self.assertEqual(info['is_success'], count == 6)
            self.assertEqual(info['collected_waste'], item)
            self.assertEqual(info['collected_count'], count)
            self.assertEqual(info['remaining_waste'], 6 - count)
            self.assertAlmostEqual(info['collection_fraction'], count / 6)
            self.assertNotIn(item, seen)
            seen.add(item)
            self.assertGreater(env.steps, previous_steps)
            previous_steps = env.steps
            self.assertGreater(env.model.body_pos[env.waste_body_ids[item], 0], 10)
            self.assertEqual(info['team_reward'], 10 if count == 6 else 0)
            for name, robot in env.robots.items():
                self.assertFalse(robot.completed)
                self.assertEqual(robot.detected_type, -1)
                np.testing.assert_array_equal(observations[name][10:13], 0)
                hits = robot.ray_geomids[robot.ray_geomids >= 0]
                self.assertFalse(np.any(env.geom_waste_ids[hits] == item))
        self.assertEqual(len(seen), 6)
        np.testing.assert_array_equal(snapshot['waste_collected'], False)
        _, rewards, terminated, _, repeated = ma_env.step_all(actions)
        self.assertTrue(terminated)
        self.assertFalse(repeated['collected'])
        self.assertEqual(repeated['team_reward'], 0)
        self.assertTrue(all(reward == 0 for reward in rewards.values()))

    def test_report_handoff_and_material_memory_are_scoped_to_the_current_item(self):
        ma_env = self.make_env()
        env = ma_env.env
        env.current_waste = 2  # First cardboard item.
        actions = self.place_current(ma_env, nearest=0)
        observations, _, terminated, _, info = ma_env.step_all(actions)
        self.assertFalse(terminated)
        self.assertEqual(info['reported_waste'], 2)
        self.assertEqual(info['collected_count'], 0)
        self.assertEqual(info['active_robot'], 'robot_2')
        self.assertEqual(info['known_waste_types'][2], 1)
        np.testing.assert_array_equal(np.delete(info['known_waste_types'], 2), -1)
        np.testing.assert_array_equal(observations['robot_2'][10:13], [0, 1, 0])
        collector = env.robots['robot_2']
        qpos = env.data.qpos.copy()
        qpos[env.robot.qpos_indices] = [-2.7, -2.5]
        qpos[collector.qpos_indices] = [1.0, 0]
        env.set_state(qpos, np.zeros(env.model.nv))
        observations, _, terminated, _, info = ma_env.step_all(actions)
        self.assertFalse(terminated)
        self.assertEqual(info['collected_waste'], 2)
        self.assertNotEqual(info['current_waste'], 2)
        np.testing.assert_array_equal(info['shared_comm'], 0)
        for observation in observations.values():
            np.testing.assert_array_equal(observation[10:13], 0)

    def test_unassigned_waste_does_not_identify_the_target_or_fail_on_proximity(self):
        ma_env = self.make_env()
        env = ma_env.env
        env.current_waste = 0  # Plastic target, hidden behind the robot.
        actions = self.place_current(ma_env, nearest=0)
        env.target[:] = [-2, 0]
        env.waste_positions[0] = env.target
        env.model.body_pos[env.waste_body_ids[0], :2] = env.target
        # A cardboard item is visible; residual waste is inside the collection radius.
        env.model.body_pos[env.waste_body_ids[2], :2] = [1.2, -0.1]
        env.model.body_pos[env.waste_body_ids[4], :2] = [0, 0]
        env.set_state(env.data.qpos.copy(), np.zeros(env.model.nv))
        _, _, terminated, _, info = ma_env.step_all(actions)
        self.assertIn(1, env.robot.ray_types)
        self.assertEqual(env.robot.detected_type, -1)
        self.assertFalse(info['reported'])
        self.assertFalse(info['collected'])
        self.assertFalse(info['wrong_target'])
        self.assertFalse(terminated)

    def test_shared_geometry_groups_do_not_hide_teammates_or_leave_mutated_groups(self):
        ma_env = self.make_env(robots=(2, 2, 2))
        env = ma_env.env
        robots = list(env.robots.values())
        original_groups = env.model.geom_group.copy()
        for body_id in env.waste_body_ids:
            env.model.body_pos[body_id, :2] = [20, 20]
        # These robot pairs share group 1, including every extra robot after the third.
        for own, other in permutations([robots[0], robots[3], robots[5]], 2):
            qpos = env.data.qpos.copy()
            for index, robot in enumerate(robots):
                qpos[robot.qpos_indices] = [-3, -2.5 + index]
                env.model.body_quat[robot.visual_body_id] = [1, 0, 0, 0]
            qpos[own.qpos_indices] = [0, 0]
            qpos[other.qpos_indices] = [1, 0]
            env.set_state(qpos, np.zeros(env.model.nv))
            own.update_rays()
            self.assertEqual(own.ray_types[1], 4)
            self.assertIn(own.ray_geomids[1], other.geom_ids)
            hits = own.ray_geomids[own.ray_geomids >= 0]
            self.assertFalse(np.any(np.isin(hits, own.geom_ids)))
            np.testing.assert_array_equal(env.model.geom_group, original_groups)
            qpos[other.qpos_indices] = [0.3, 0]
            env.set_state(qpos, np.zeros(env.model.nv))
            own.go = 1
            self.assertLessEqual(own.compute_avoidance_reward(), -own.robot_collision_penalty)
        with patch('raycast.mujoco.mj_multiRay', side_effect=RuntimeError('ray failure')):
            with self.assertRaises(RuntimeError):
                own.update_rays()
        np.testing.assert_array_equal(env.model.geom_group, original_groups)

    def test_six_robots_share_one_policy_and_timeout_reports_partial_progress(self):
        env = self.make_env(robots=(2, 2, 2))
        model = Mock()
        model.predict.side_effect = lambda observations: (np.zeros((len(observations), 2)), None)
        env.env.max_steps = 2
        with redirect_stdout(StringIO()):
            ma_evaluate(env, {'robot': model}, total_episodes=1)
        self.assertEqual(model.predict.call_count, 2)
        self.assertEqual(model.predict.call_args.args[0].shape, (6, 76))
        self.assertEqual(env.episode_successes, [False])
        self.assertEqual(env.episode_collection_fractions, [0.0])

    def test_single_team_and_default_counts_work(self):
        env = self.make_env(robots=(0, 0, 1), waste=(0, 0, 2))
        self.assertEqual(env.env.robot.specialization, 2)
        self.assertEqual(env.env.robot_count, 1)
        self.assertEqual(env.env.active_robot, 'robot')
        default = RobotMAEnv(sorting=True, robots_per_team=(2, 1, 0))
        self.addCleanup(default.close)
        _, info = default.reset(seed=0)
        self.assertEqual(default.env.waste_per_type, (1, 1, 0))
        self.assertEqual(info['total_waste'], 2)

    def test_timeout_preserves_partial_collection_without_paying_the_final_team_bonus(self):
        ma_env = self.make_env(robots=(1, 0, 0), waste=(2, 0, 0))
        env = ma_env.env
        actions = self.place_current(ma_env, nearest=0)
        ma_env.step_all(actions)
        qpos = env.data.qpos.copy()
        qpos[env.robot.qpos_indices] = [1.0, 0]
        env.set_state(qpos, np.zeros(env.model.nv))
        _, _, terminated, _, info = ma_env.step_all(actions)
        self.assertFalse(terminated)
        self.assertEqual(info['collected_count'], 1)
        env.max_steps = env.steps + 1
        _, _, terminated, truncated, info = ma_env.step_all(actions)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertFalse(info['is_success'])
        self.assertEqual(info['collection_fraction'], 0.5)
        self.assertEqual(info['remaining_waste'], 1)
        self.assertEqual(info['team_reward'], 0)
        self.assertEqual(ma_env.episode_collection_fractions, [0.5])

    def test_initial_seed_drives_a_repeatable_sequence_of_distinct_episodes(self):
        first = RobotMAEnv(sorting=True, robots_per_team=(1, 1, 1), seed=17)
        second = RobotMAEnv(sorting=True, robots_per_team=(1, 1, 1))
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        previous = None
        for episode in range(3):
            obs_a, info_a = first.reset()
            obs_b, info_b = second.reset(seed=17 if episode == 0 else None)
            np.testing.assert_array_equal(info_a['waste_positions'], info_b['waste_positions'])
            for name in obs_a:
                np.testing.assert_array_equal(obs_a[name], obs_b[name])
            if previous is not None:
                self.assertFalse(np.array_equal(previous, info_a['waste_positions']))
            previous = info_a['waste_positions']

    def test_invalid_or_unsolvable_configurations_are_rejected(self):
        configurations = [
            {'robots_per_team': (1, 1, 1)},
            {'sorting': True, 'robots_per_team': (0, 0, 0)},
            {'sorting': True, 'robots_per_team': (-1, 1, 1)},
            {'sorting': True, 'robots_per_team': (1, 1)},
            {'sorting': True, 'waste_per_type': (0, 0, 0)},
            {'sorting': True, 'robots_per_team': (1, 0, 0), 'waste_per_type': (1, 1, 0)},
            {'sorting': True, 'robots_per_team': (1, 1, 1), 'drone_search': True},
            {'sorting': True, 'robots_per_team': (1, 1, 1), 'three_robots': True},
        ]
        for kwargs in configurations:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                WasteCollectionEnv(**kwargs)


if __name__ == '__main__':
    unittest.main()
