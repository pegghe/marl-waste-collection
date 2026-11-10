"""Penalize confined search oscillations without penalizing useful wide loops."""

import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from waste_collection_env import WasteCollectionEnv


class DroneOscillationTest(unittest.TestCase):
    def setUp(self):
        self.env = WasteCollectionEnv(drone_training=True, finish_on_discovery=False)
        self.addCleanup(self.env.close)
        self.env.reset(seed=42)
        self.drone = self.env.drone
        self.start_at([0.25, 0.25])
        # Isolate search behavior from incidental ray hits along test paths.
        rays = patch.object(self.drone, "update_rays")
        rays.start()
        self.addCleanup(rays.stop)

    def start_at(self, position):
        self.env.data.mocap_pos[self.drone.mocap_id, :2] = position
        mujoco.mj_forward(self.env.model, self.env.data)
        self.drone.reset()

    def step(self, action, detected=False):
        self.drone.is_bottle[:] = detected
        self.env.move_drone(np.asarray(action))
        return self.env.step(np.zeros(2))[-1]

    def oscillate(self, steps, first_direction=0.4):
        info = None
        for index in range(steps):
            direction = first_direction if index % 2 == 0 else -first_direction
            info = self.step([direction, 0])
        return info

    def test_fast_small_oscillations_are_penalized_after_full_window(self):
        for index in range(49):
            info = self.step([0.4 if index % 2 == 0 else -0.4, 0])
            self.assertEqual(info["drone_reward_components"]["oscillation"], 0)
        info = self.step([-0.4, 0])
        self.assertAlmostEqual(info["drone_speed"], 0.4)
        self.assertEqual(info["drone_reward_components"]["stagnation"], 0)
        self.assertEqual(info["drone_reward_components"]["oscillation"], -0.05)
        self.assertAlmostEqual(info["drone_reward"], -0.1)
        self.assertEqual(self.step([0.4, 0])["drone_reward_components"]["oscillation"], -0.05)

    def test_wide_closed_loop_is_not_oscillation_even_without_new_cells(self):
        self.drone.visited_map.fill(1)
        actions = [[1, 0]] * 20 + [[0, 1]] * 5 + [[-1, 0]] * 20 + [[0, -1]] * 5
        for action in actions:
            info = self.step(action)
        np.testing.assert_allclose(
            self.env.data.xpos[self.drone.body_id, :2], [0.25, 0.25], atol=1e-10,
        )
        self.assertEqual(info["drone_reward_components"]["exploration"], 0)
        self.assertEqual(info["drone_reward_components"]["oscillation"], 0)

    def test_new_cell_protects_the_entire_recent_window(self):
        self.start_at([0.49, 0.25])
        self.oscillate(49, first_direction=-0.4)
        info = self.step([1, 0])
        self.assertEqual(info["drone_reward_components"]["exploration"], 0.5)
        self.assertEqual(info["drone_reward_components"]["oscillation"], 0)
        for index in range(49):
            info = self.step([0.4 if index % 2 == 0 else -0.4, 0])
            self.assertEqual(info["drone_reward_components"]["exploration"], 0)
            self.assertEqual(info["drone_reward_components"]["oscillation"], 0)
        info = self.step([-0.4, 0])
        self.assertEqual(info["drone_reward_components"]["oscillation"], -0.05)

    def test_leaving_confined_region_removes_penalty_without_a_new_cell(self):
        self.drone.visited_map.fill(1)
        self.assertEqual(self.oscillate(50)["drone_reward_components"]["oscillation"], -0.05)
        for _ in range(7):
            info = self.step([1, 0])
        self.assertEqual(info["drone_reward_components"]["exploration"], 0)
        self.assertEqual(info["drone_reward_components"]["oscillation"], 0)

    def test_discovery_and_later_steps_do_not_charge_oscillation(self):
        self.oscillate(50)
        info = self.step([0.4, 0], detected=True)
        self.assertGreater(info["drone_reward_components"]["discovery"], 0)
        self.assertEqual(info["drone_reward_components"]["oscillation"], 0)
        self.assertEqual(self.step([-0.4, 0])["drone_reward"], 0)

    def test_reset_clears_history_and_preserves_checkpoint_observation(self):
        self.assertEqual(self.oscillate(50)["drone_reward_components"]["oscillation"], -0.05)
        self.env.reset(seed=43)
        self.assertTrue(all(value == 0 for value in self.drone.reward_components.values()))
        self.drone.visited_map.fill(1)
        for index in range(49):
            # Reading observations must not advance the search window.
            for _ in range(3):
                observation = self.drone.get_observation()
                self.assertEqual(observation.shape, (219,))
                self.assertTrue(self.drone.observation_space.contains(observation))
            info = self.step([0.4 if index % 2 == 0 else -0.4, 0])
            self.assertEqual(info["drone_reward_components"]["oscillation"], 0)
        self.assertEqual(self.step([-0.4, 0])["drone_reward_components"]["oscillation"], -0.05)


if __name__ == "__main__":
    unittest.main()
