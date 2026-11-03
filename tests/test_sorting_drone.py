"""The drone locates either material before dispatching the nearest robot."""

import unittest

import numpy as np

from waste_collection_env import RobotMAEnv


class SortingDroneTest(unittest.TestCase):
    def setUp(self):
        self.ma_env = RobotMAEnv(sorting=True, two_robots=True, drone_search=True)
        self.addCleanup(self.ma_env.close)
        self.ma_env.reset(seed=42)
        self.env = self.ma_env.env
        self.actions = {name: np.zeros(2) for name in self.ma_env.agents}

    def place(self, material):
        self.ma_env.reset(seed=42)
        env = self.env
        env.target_type = material
        env.target = np.array([1.2, 0])
        for kind, body in enumerate(('target', 'paper')):
            env.model.body(body).pos[:2] = env.target if kind == material else [20, 20]
        qpos = env.data.qpos.copy()
        qpos[env.robot.qpos_indices] = [-2, -2]
        qpos[env.robots['robot_2'].qpos_indices] = [-2, 2]
        env.data.mocap_pos[env.drone.mocap_id] = [-3, 3, 3]
        env.set_state(qpos, np.zeros(env.model.nv))
        env.assign_robot()

    def test_both_materials_are_detected_without_broadcasting_their_type(self):
        for material in (0, 1):
            with self.subTest(material=material):
                self.place(material)
                start = {name: robot.position.copy() for name, robot in self.env.robots.items()}
                observations, _, terminated, _, info = self.ma_env.step_all(
                    {**self.actions, 'robot': [1, 1], 'robot_2': [1, 1]})
                self.assertFalse(terminated)
                self.assertFalse(info['drone_found_target'])
                self.assertIsNone(info['active_robot'])
                for name, robot in self.env.robots.items():
                    np.testing.assert_array_equal(robot.position, start[name])
                    np.testing.assert_array_equal(observations[name][4:7], 0)
                self.env.data.mocap_pos[self.env.drone.mocap_id, :2] = self.env.target
                observations, _, terminated, _, info = self.ma_env.step_all(self.actions)
                self.assertFalse(terminated)
                self.assertTrue(info['drone_found_target'])
                self.assertFalse(info['reported'])
                np.testing.assert_allclose(info['shared_comm'], [1, 1.2 / 5, 0, 0])
                self.assertEqual(observations['drone'].shape, (219,))
                self.assertTrue(self.env.drone_observation_space.contains(observations['drone']))
                self.assertTrue(np.any(observations['drone'][5:23].reshape(9, 2)[:, 1]))
                self.assertEqual(info['active_robot'], 'robot')
                for name, robot in self.env.robots.items():
                    np.testing.assert_array_equal(robot.position, start[name])
                    np.testing.assert_allclose(observations[name][4:6], self.env.target - robot.position)
                    np.testing.assert_array_equal(observations[name][10:13], 0)
                    self.assertEqual(observations[name][6], int(name == 'robot'))

    def test_drone_location_then_robot_report_then_compatible_collection(self):
        self.place(1)
        env = self.env
        env.data.mocap_pos[env.drone.mocap_id, :2] = env.target
        self.ma_env.step_all(self.actions)
        qpos = env.data.qpos.copy()
        qpos[env.robot.qpos_indices] = [0, 0]
        env.set_state(qpos, np.zeros(env.model.nv))
        _, _, terminated, _, info = self.ma_env.step_all(self.actions)
        self.assertFalse(terminated)
        self.assertTrue(info['reported'])
        self.assertEqual(info['active_robot'], 'robot_2')
        self.assertEqual(info['shared_comm'][3], 2)
        self.assertEqual(env.robots['robot_2'].detected_type, 1)
        drone_position = env.data.mocap_pos[env.drone.mocap_id].copy()
        env.move_drone([1, 1])
        np.testing.assert_array_equal(env.data.mocap_pos[env.drone.mocap_id], drone_position)
        qpos[env.robot.qpos_indices] = [-3, -2]
        qpos[env.robots['robot_2'].qpos_indices] = [1.0, 0]
        env.set_state(qpos, np.zeros(env.model.nv))
        _, _, terminated, _, info = self.ma_env.step_all(self.actions)
        self.assertTrue(terminated)
        self.assertTrue(info['collected'])
        self.assertTrue(info['is_success'])
        self.assertEqual(info['team_reward'], 10)
        self.assertEqual(info['shared_comm'][3], 2)


if __name__ == '__main__':
    unittest.main()
