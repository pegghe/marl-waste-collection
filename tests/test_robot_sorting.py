"""A random specialist collects matching waste or reports incompatible waste."""

import unittest

import mujoco
import numpy as np

from waste_collection_env import WasteCollectionEnv


class RobotSortingTest(unittest.TestCase):
    def setUp(self):
        self.env = WasteCollectionEnv(sorting=True)
        self.addCleanup(self.env.close)
        self.env.reset(seed=42)

    def place(self, specialization, material, position=(0, 0), target=(1.2, 0)):
        env = self.env
        robot = env.robot
        robot.specialization = specialization
        robot.detected_type = -1
        robot.completed = False
        robot.go = 1
        robot.go_delay = 0
        env.target_type = material
        env.target = np.asarray(target, dtype=float)
        env.shared_comm.fill(0)
        env.model.body_quat[robot.visual_body_id] = [1, 0, 0, 0]
        for kind, body in enumerate(('target', 'paper')):
            env.model.body(body).pos[:2] = target if kind == material else [20, 20]
        qpos = env.data.qpos.copy()
        qpos[robot.qpos_indices] = position
        env.set_state(qpos, np.zeros(env.model.nv))
        robot.update_rays()

    def test_roles_and_waste_are_independent_and_reset_is_repeatable(self):
        cases = set()
        for seed in range(64):
            observation, _ = self.env.reset(seed=seed)
            cases.add((self.env.robot.specialization, self.env.target_type))
            self.assertGreaterEqual(np.linalg.norm(self.env.target - self.env.robot.position), 3.0)
            self.assertEqual(self.env.robot.detected_type, -1)
            self.assertEqual(observation.shape, (65,))
            self.assertTrue(self.env.observation_space.contains(observation))
            repeated, _ = self.env.reset(seed=seed)
            np.testing.assert_array_equal(observation, repeated)
            np.testing.assert_array_equal(self.env.shared_comm, 0)
        self.assertEqual(cases, {(0, 0), (0, 1), (1, 0), (1, 1)})

    def test_observation_contains_role_and_ray_types_without_revealing_unseen_material(self):
        observations = []
        for material in (0, 1):
            self.place(0, material, target=(-1.2, 0))
            observation = self.env._get_obs()
            np.testing.assert_array_equal(observation[7:9], [1, 0])
            np.testing.assert_array_equal(observation[9:11], [0, 0])
            rays = observation[11:].reshape(9, 6)
            np.testing.assert_array_equal(rays[:, 0], 1)
            np.testing.assert_array_equal(rays[:, 5], 1)  # No object.
            np.testing.assert_array_equal(rays[:, 1:].sum(axis=1), 1)
            observations.append(observation)
        np.testing.assert_array_equal(*observations)
        for material in (0, 1):
            self.place(material, material)
            self.assertIn(material, self.env.robot.ray_types)
            self.assertEqual(self.env.robot.detected_type, material)
            self.assertTrue(self.env._get_obs()[9 + material])

    def test_both_specializations_collect_only_after_identifying_matching_waste(self):
        for material in (0, 1):
            with self.subTest(material=material):
                self.place(material, material)
                _, _, terminated, _, _ = self.env.step(np.zeros(2))
                self.assertFalse(terminated)
                # Retain recognition when the object passes behind the head sensor.
                qpos = self.env.data.qpos.copy()
                qpos[self.env.robot.qpos_indices] = [1.0, 0]
                self.env.set_state(qpos, np.zeros(self.env.model.nv))
                _, reward, terminated, truncated, info = self.env.step(np.zeros(2))
                self.assertTrue(terminated)
                self.assertFalse(truncated)
                self.assertTrue(info['is_success'])
                self.assertTrue(info['collected'])
                self.assertFalse(info['reported'])
                self.assertGreater(reward, 0)

    def test_both_specializations_report_wrong_material_outside_collection_radius(self):
        for specialization, material in ((0, 1), (1, 0)):
            with self.subTest(specialization=specialization):
                self.place(specialization, material)
                _, reward, terminated, truncated, info = self.env.step(np.zeros(2))
                self.assertTrue(terminated)
                self.assertFalse(truncated)
                self.assertTrue(info['is_success'])
                self.assertTrue(info['reported'])
                self.assertFalse(info['collected'])
                self.assertGreater(info['distance'], 0.25)
                np.testing.assert_allclose(info['shared_comm'], [1, 1.2 / 5, 0, material + 1])
                self.assertGreater(reward, 0)
                self.assertEqual(self.env.robot.compute_reward()[0], 0)
                np.testing.assert_array_equal(self.env.robot.control([1, 1]), 0)
                self.env.reset(seed=42)
                np.testing.assert_array_equal(self.env.shared_comm, 0)
                # The message in the completed episode is an independent snapshot.
                self.assertEqual(info['shared_comm'][3], material + 1)

    def test_unseen_wrong_material_is_not_reported(self):
        self.place(0, 1, target=(-1.2, 0))
        _, _, terminated, _, info = self.env.step(np.zeros(2))
        self.assertFalse(terminated)
        self.assertFalse(info['reported'])
        np.testing.assert_array_equal(info['shared_comm'], 0)

    def test_entering_wrong_collection_radius_is_a_failure_not_a_collection(self):
        for specialization, material in ((0, 1), (1, 0)):
            with self.subTest(specialization=specialization):
                self.place(specialization, material, position=(1.0, 0))
                _, reward, terminated, truncated, info = self.env.step(np.zeros(2))
                self.assertTrue(terminated)
                self.assertFalse(truncated)
                self.assertFalse(info['is_success'])
                self.assertTrue(info['wrong_target'])
                self.assertFalse(info['collected'])
                self.assertLess(reward, -10)

    def test_go_signal_still_controls_task_completion(self):
        self.place(0, 1)
        self.env.robot.go = 0
        self.env.robot.go_delay = 100
        _, _, terminated, _, info = self.env.step(np.zeros(2))
        self.assertFalse(terminated)
        self.assertFalse(info['reported'])
        self.env.steps = 100
        self.env.step(np.zeros(2))  # Make go available to the next action.
        _, _, terminated, _, info = self.env.step(np.zeros(2))
        self.assertTrue(terminated)
        self.assertTrue(info['reported'])

    def test_non_waste_geometry_is_an_obstacle(self):
        self.place(0, 0, target=(-1.2, 0))
        # Put an existing non-waste body in front of the horizontal ray.
        self.env.data.mocap_pos[self.env.drone.mocap_id] = [1, 0, 0.35]
        mujoco.mj_forward(self.env.model, self.env.data)
        self.env.robot.update_rays()
        self.assertEqual(self.env.robot.ray_types[1], 2)
        self.assertEqual(self.env.robot.detected_type, -1)


if __name__ == '__main__':
    unittest.main()
