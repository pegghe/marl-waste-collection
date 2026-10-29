"""The robot learns the target only through the drone's broadcast."""
import unittest
from unittest.mock import patch
import numpy as np
from waste_collection_env import WasteCollectionEnv


class TargetCommunicationTest(unittest.TestCase):
    def test_target_hidden_until_detection_and_robot_can_finish(self):
        env = WasteCollectionEnv(drone_training=True, finish_on_discovery=False)
        try:
            observation, _ = env.reset(seed=42)
            np.testing.assert_array_equal(observation[4:], [0, 0, 0])
            # Waiting beyond the old random delay must not reveal the target.
            env.steps = 150
            env.drone.is_bottle = np.zeros(9, dtype=bool)
            env.drone.compute_reward()
            env.robot.compute_reward()
            np.testing.assert_array_equal(env._get_obs()[4:], [0, 0, 0])
            env.data.mocap_pos[env.drone.mocap_id, :2] = env.target
            observation, _, terminated, truncated, info = env.step(np.zeros(2))
            self.assertTrue(info['drone_found_target'])
            self.assertFalse(terminated or truncated)
            self.assertEqual(observation[-1], 1)
            np.testing.assert_allclose(observation[4:6], env.target-env.data.qpos[:2], atol=1e-6)
            # Communication persists after the drone moves away.
            env.data.mocap_pos[env.drone.mocap_id, :2] = [5, 5]
            env.step(np.zeros(2))
            self.assertEqual(env._get_obs()[-1], 1)
            qpos = env.data.qpos.copy()
            qpos[:2] = env.target
            env.set_state(qpos, np.zeros_like(env.data.qvel))
            _, _, terminated, _, info = env.step(np.zeros(2))
            self.assertTrue(terminated)
            self.assertTrue(info['is_success'])
            observation, _ = env.reset(seed=43)
            np.testing.assert_array_equal(observation[4:], [0, 0, 0])
            np.testing.assert_array_equal(env.shared_comm, [0, 0, 0])
        finally:
            env.close()

    def test_robot_actions_blocked_until_broadcast_and_after_reset(self):
        env = WasteCollectionEnv(drone_training=True, finish_on_discovery=False)
        try:
            env.reset(seed=42)
            start = env.data.qpos[:2].copy()
            # Keep search active while feeding a policy that constantly commands motion.
            with patch.object(env.drone, 'update_rays'):
                env.drone.is_bottle[:] = False
                for _ in range(120):
                    env.step(np.ones(2))
            np.testing.assert_allclose(env.data.qpos[:2], start, atol=1e-10)
            np.testing.assert_array_equal(env.data.ctrl, [0, 0])
            # Detect through actual rays; actions on the detection step remain zero.
            env.data.mocap_pos[env.drone.mocap_id, :2] = env.target
            observation, *_ = env.step(np.ones(2))
            self.assertEqual(observation[-1], 1)
            np.testing.assert_array_equal(env.data.ctrl, [0, 0])
            env.step(np.ones(2))
            np.testing.assert_array_equal(env.data.ctrl, [1, 1])
            self.assertGreater(np.linalg.norm(env.data.qpos[:2] - start), 0)
            env.reset(seed=43)
            env.robot.apply_action(np.ones(2))
            np.testing.assert_array_equal(env.data.ctrl, [0, 0])
        finally:
            env.close()

    def test_robot_only_mode_keeps_policy_control(self):
        env = WasteCollectionEnv()
        try:
            env.reset(seed=42)
            env.robot.apply_action(np.ones(2))
            np.testing.assert_array_equal(env.data.ctrl, [1, 1])
        finally:
            env.close()

    def test_drone_training_still_ends_at_detection(self):
        env = WasteCollectionEnv(drone_training=True)
        try:
            env.reset(seed=42)
            env.data.mocap_pos[env.drone.mocap_id, :2] = env.target
            observation, _, terminated, _, info = env.step(np.zeros(2))
            self.assertTrue(terminated)
            self.assertTrue(info['drone_found_target'])
            self.assertEqual(observation[-1], 1)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
