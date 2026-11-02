"""Two independent robots, shared physics and head-mounted rays."""

import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from waste_collection_env import RobotMAEnv, WasteCollectionEnv


class RobotHeadRaysTest(unittest.TestCase):
    def setUp(self):
        self.env = WasteCollectionEnv(two_robots=True)
        self.addCleanup(self.env.close)
        self.env.reset(seed=42)

    def place(self, first, second, target):
        env = self.env
        qpos = env.data.qpos.copy()
        qpos[env.robot.qpos_indices] = first
        qpos[env.robots['robot_2'].qpos_indices] = second
        env.target = np.asarray(target, dtype=float)
        env.model.body('target').pos[:2] = target
        env.set_state(qpos, np.zeros(env.model.nv))
        for robot in env.robots.values():
            robot.update_rays()

    def test_both_spawns_are_repeatable_and_clear(self):
        first, _ = self.env.reset(seed=17)
        positions = [robot.position.copy() for robot in self.env.robots.values()]
        second, info = self.env.reset(seed=17)
        np.testing.assert_array_equal(first, second)
        for robot, expected in zip(self.env.robots.values(), positions):
            np.testing.assert_array_equal(robot.position, expected)
            self.assertGreaterEqual(np.linalg.norm(robot.position - self.env.target), 0.75)
        self.assertGreaterEqual(np.linalg.norm(positions[0] - positions[1]), 1.0)
        self.assertEqual(set(info['robot_sensors']), {'robot', 'robot_2'})

    def test_each_head_detects_plastic_and_ignores_its_own_shell(self):
        for name in self.env.robots:
            with self.subTest(robot=name):
                if name == 'robot':
                    self.place([0, 0], [-2, 2], [1.2, 0])
                else:
                    self.place([-2, 2], [0, 0], [1.2, 0])
                robot = self.env.robots[name]
                self.assertIn(robot.ray_geomids[4], self.env.drone.target_geom_ids)
                hit_ids = robot.ray_geomids[robot.ray_geomids >= 0]
                own_group = 1 if name == 'robot' else 2
                self.assertFalse(np.any(self.env.model.geom_group[hit_ids] == own_group))

    def test_heads_follow_yaw_and_detect_other_robot(self):
        for name in self.env.robots:
            with self.subTest(robot=name):
                if name == 'robot':
                    self.place([0, 0], [1, 0], [-2, -2])
                else:
                    self.place([1, 0], [0, 0], [-2, -2])
                robot = self.env.robots[name]
                # The horizontal center ray must hit the other robot.
                geom_id = robot.ray_geomids[1]
                self.assertGreaterEqual(geom_id, 0)
                other_group = 2 if name == 'robot' else 1
                self.assertEqual(self.env.model.geom_group[geom_id], other_group)
                self.assertEqual(robot.ray_types[1], 3)
                origin = robot.raycaster.raycast_origins_buf[robot.body_id]
                np.testing.assert_allclose(origin[:2], robot.position + [0.35, 0], atol=1e-5)
                self.assertAlmostEqual(origin[2], 0.35, places=5)
                # Turn the visual shell/head 90 degrees; the center ray must turn too.
                self.env.model.body_quat[robot.visual_body_id] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
                mujoco.mj_forward(self.env.model, self.env.data)
                robot.update_rays()
                origin = robot.raycaster.raycast_origins_buf[robot.body_id]
                direction = robot.raycaster.raycast_ray_dirs_buf[robot.body_id][4]
                np.testing.assert_allclose(origin[:2], robot.position + [0, 0.35], atol=1e-5)
                self.assertAlmostEqual(direction[0], 0, places=6)
                self.assertGreater(direction[1], 0)
                self.assertLess(direction[2], 0)
                # Restore the pose before testing the other robot.
                self.env.model.body_quat[robot.visual_body_id] = [1, 0, 0, 0]

    def test_empty_rays_and_info_are_independent_snapshots(self):
        self.place([0, 0], [-2, 2], [-2, -2])
        robot = self.env.robot
        np.testing.assert_array_equal(robot.ray_geomids, -1)
        np.testing.assert_array_equal(robot.ray_distances, -1)
        sensors = self.env.get_robot_sensors()
        sensors['robot']['geomids'][:] = 0
        sensors['robot']['distances'][:] = 0
        np.testing.assert_array_equal(robot.ray_geomids, -1)
        np.testing.assert_array_equal(robot.ray_distances, -1)

    def test_independent_controls_advance_physics_only_once(self):
        env = RobotMAEnv(two_robots=True)
        self.addCleanup(env.close)
        observations, _ = env.reset(seed=42)
        self.assertEqual(set(observations), {'robot', 'robot_2'})
        self.assertTrue(all(obs.shape == (7,) for obs in observations.values()))
        for agent in env.agents.values():
            self.assertEqual(agent.action_space.shape, (2,))
            self.assertEqual(agent.model_name, 'robot')
        start = env.env.data.time
        with patch.object(env.env, 'do_simulation', wraps=env.env.do_simulation) as simulate:
            _, rewards, *_ = env.step_all({'robot': [1, 0], 'robot_2': [0, -1]})
        self.assertEqual(simulate.call_count, 1)
        self.assertAlmostEqual(env.env.data.time - start, 0.05)
        np.testing.assert_array_equal(env.env.data.ctrl, [1, 0, 0, -1])
        self.assertEqual(set(rewards), {'robot', 'robot_2'})
        self.assertGreater(env.env.robot.velocity[0], 0)
        self.assertLess(env.env.robots['robot_2'].velocity[1], 0)

    def test_broadcast_releases_both_robots_and_reset_clears_actions(self):
        env = RobotMAEnv(two_robots=True, drone_training=True, finish_on_discovery=False)
        self.addCleanup(env.close)
        observations, _ = env.reset(seed=42)
        for name in ('robot', 'robot_2'):
            np.testing.assert_array_equal(observations[name][4:], [0, 0, 0])
        env.step_all({'robot': [1, 0], 'robot_2': [0, -1], 'drone': [0, 0]})
        np.testing.assert_array_equal(env.env.data.ctrl, 0)
        env.env.shared_comm[:] = [1, *env.env.target / 5]
        for robot in env.env.robots.values():
            observation = robot.get_observation()
            np.testing.assert_allclose(observation[4:6], env.env.target - robot.position)
            self.assertEqual(observation[6], 1)
        env.step_all({'robot': [1, 0], 'robot_2': [0, -1], 'drone': [0, 0]})
        np.testing.assert_array_equal(env.env.data.ctrl, [1, 0, 0, -1])
        env.reset(seed=43)
        np.testing.assert_array_equal(env.action, 0)
        np.testing.assert_array_equal(env.drone_action, 0)

    def test_each_robot_finishes_once_while_other_can_continue(self):
        env = self.env
        self.place([0, 0], [-2, 2], [0, 0])
        env.robot.go = 1
        reward, success, _ = env.robot.compute_reward()
        self.assertTrue(success)
        self.assertAlmostEqual(reward, 9.99)
        self.assertEqual(env.robot.compute_reward()[0], 0)
        np.testing.assert_array_equal(env.robot.control([1, 1]), 0)
        self.assertFalse(env.robots['robot_2'].completed)
        # Reaching is remembered even if another body later displaces the robot.
        self.place([-2, -2], [0, 0], [0, 0])
        env.robots['robot_2'].go = 1
        _, _, terminated, _, info = env.step(np.zeros(4))
        self.assertTrue(terminated)
        self.assertTrue(all(info['robot_successes'].values()))
        env.reset(seed=42)
        self.assertTrue(all(not r.completed for r in env.robots.values()))


if __name__ == '__main__':
    unittest.main()
