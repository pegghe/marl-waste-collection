"""Actual drone progress, boundary costs, and search-only penalties."""

import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from waste_collection_env import WasteCollectionEnv


class DroneMovementTest(unittest.TestCase):
    def setUp(self):
        self.env = WasteCollectionEnv(drone_training=True, finish_on_discovery=False)
        self.addCleanup(self.env.close)
        self.env.reset(seed=42)

    def step_from(self, position, action, detected=False):
        drone = self.env.drone
        self.env.data.mocap_pos[drone.mocap_id, :2] = position
        mujoco.mj_forward(self.env.model, self.env.data)
        drone.reset()
        with patch.object(drone, "update_rays"):
            drone.is_bottle[:] = detected
            self.env.move_drone(np.asarray(action))
            return self.env.step(np.zeros(2))[-1]

    def test_useful_movement_keeps_original_speed_without_extra_penalties(self):
        info = self.step_from([0.25, 0.25], [1, 0])
        np.testing.assert_allclose(
            self.env.data.mocap_pos[self.env.drone.mocap_id, :2], [0.3, 0.25],
        )
        self.assertAlmostEqual(info["drone_speed"], 1)
        self.assertEqual(info["drone_reward_components"]["boundary"], 0)
        self.assertEqual(info["drone_reward_components"]["stagnation"], 0)

    def test_edge_penalty_charges_outward_commands_but_allows_return_and_sliding(self):
        blocked = self.step_from([3.5, 0.25], [1, 0])
        self.assertAlmostEqual(blocked["drone_reward_components"]["boundary"], -0.2)
        self.assertAlmostEqual(blocked["drone_reward"], -0.3)
        self.assertEqual(blocked["drone_speed"], 0)
        self.assertAlmostEqual(blocked["drone_blocked_fraction"], 1)
        for action in ([-1, 0], [0, 1]):
            info = self.step_from([3.5, 0.25], action)
            self.assertEqual(info["drone_reward_components"]["boundary"], 0)
            self.assertEqual(info["drone_reward_components"]["stagnation"], 0)
            self.assertAlmostEqual(info["drone_reward"], -0.05)

    def test_partial_clipping_cost_is_proportional_and_corner_stays_bounded(self):
        partial = self.step_from([3.475, 0.25], [1, 0])
        self.assertAlmostEqual(partial["drone_blocked_fraction"], 0.5)
        self.assertAlmostEqual(partial["drone_reward_components"]["boundary"], -0.1)
        corner = self.step_from([3.5, 3.5], [1, 1])
        self.assertAlmostEqual(corner["drone_reward_components"]["boundary"], -0.2)
        np.testing.assert_array_equal(
            self.env.data.mocap_pos[self.env.drone.mocap_id, :2], [3.5, 3.5],
        )

    def test_stagnation_cost_decreases_smoothly_with_actual_speed(self):
        costs = [self.step_from([0.25, 0.25], action)["drone_reward_components"]["stagnation"]
                 for action in ([0, 0], [0.1, 0], [1, 0])]
        self.assertAlmostEqual(costs[0], -0.05)
        self.assertLess(costs[0], costs[1])
        self.assertLess(costs[1], costs[2])
        self.assertEqual(costs[2], 0)

    def test_discovery_and_reset_remove_search_penalties(self):
        info = self.step_from([3.5, 3.5], [1, 1], detected=True)
        self.assertGreater(info["drone_reward_components"]["discovery"], 0)
        self.assertEqual(info["drone_reward_components"]["boundary"], 0)
        self.assertEqual(info["drone_reward_components"]["stagnation"], 0)
        with patch.object(self.env.drone, "update_rays"):
            self.env.move_drone(np.ones(2))
            info = self.env.step(np.zeros(2))[-1]
        self.assertEqual(info["drone_reward"], 0)
        self.env.reset(seed=43)
        self.assertEqual(self.env.drone.blocked_fraction, 0)
        self.assertEqual(self.env.drone.last_movement, 0)
        self.assertTrue(all(value == 0 for value in self.env.drone.reward_components.values()))


if __name__ == "__main__":
    unittest.main()
