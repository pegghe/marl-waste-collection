"""The drone observes its exploration history and earns each cell bonus once."""

import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from waste_collection_env import WasteCollectionEnv


class DroneExplorationMapTest(unittest.TestCase):
    def setUp(self):
        self.env = WasteCollectionEnv(drone_training=True, finish_on_discovery=False)
        self.addCleanup(self.env.close)
        self.env.reset(seed=42)

    def move_and_step(self, position):
        self.env.data.mocap_pos[self.env.drone.mocap_id, :2] = position
        # Keep search active independently of where this test places the drone.
        with patch.object(self.env.drone, "update_rays"):
            self.env.drone.is_bottle[:] = False
            return self.env.step(np.zeros(2))[-1]

    def test_reset_observation_includes_map_and_preserves_sensor_fields(self):
        _, info = self.env.reset(seed=42)
        drone = self.env.drone
        observation = info["drone_observation"]
        self.assertEqual(observation.shape, (219,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(self.env.drone_observation_space.contains(observation))
        np.testing.assert_array_equal(observation[:3], [0, 0, 0])
        np.testing.assert_allclose(
            observation[3:5], self.env.data.xpos[drone.body_id, :2] / 5,
        )
        rays = observation[5:23].reshape(9, 2)
        expected_ranges = np.where(drone.ray_distances < 0, 5, drone.ray_distances) / 5
        np.testing.assert_allclose(rays[:, 0], expected_ranges)
        np.testing.assert_array_equal(rays[:, 1], drone.is_bottle)
        grid = observation[23:].reshape(14, 14)
        np.testing.assert_array_equal(grid, info["drone_visited_map"])
        self.assertEqual(grid.sum(), 1)
        self.assertEqual(grid[drone.cell()], 1)
        # Observing the map neither changes history nor exposes a writable view.
        np.testing.assert_array_equal(drone.get_observation(), observation)
        observation[23:] = 0
        info["drone_visited_map"].fill(0)
        self.assertEqual(drone.visited_map.sum(), 1)

    def test_movement_updates_history_and_only_rewards_first_visit(self):
        drone = self.env.drone
        self.env.data.mocap_pos[drone.mocap_id, :2] = [-3.25, -3.25]
        mujoco.mj_forward(self.env.model, self.env.data)
        drone.reset()
        initial_observation = drone.get_observation()
        self.assertEqual(drone.cell(), (0, 0))

        info = self.move_and_step([-2.75, -3.25])
        self.assertEqual(drone.cell(), (0, 1))
        self.assertEqual(info["drone_reward_components"]["exploration"], 0.5)
        self.assertEqual(info["drone_reward_components"]["time"], -0.05)
        expected = np.zeros((14, 14), dtype=np.float32)
        expected[0, :2] = 1
        np.testing.assert_array_equal(info["drone_observation"][23:].reshape(14, 14), expected)
        np.testing.assert_array_equal(info["drone_visited_map"], expected)
        self.assertEqual(initial_observation[23:].sum(), 1)

        for position in ([-2.75, -3.25], [-3.25, -3.25]):
            revisit = self.move_and_step(position)
            self.assertEqual(revisit["drone_reward_components"]["exploration"], 0)
            np.testing.assert_array_equal(drone.visited_map, expected)

        self.move_and_step([-3.25, -2.75])
        self.assertEqual(drone.cell(), (1, 0))
        self.assertEqual(drone.visited_map.sum(), 3)
        # Step info is a snapshot, not a view into future exploration.
        np.testing.assert_array_equal(info["drone_visited_map"], expected)

    def test_edges_share_the_outer_cells_without_extra_exploration_bonus(self):
        for position, cell in (
            ([-3.5, -3.5], (0, 0)),
            ([3.5, -3.5], (0, 13)),
            ([-3.5, 3.5], (13, 0)),
            ([3.5, 3.5], (13, 13)),
        ):
            with self.subTest(position=position):
                info = self.move_and_step(position)
                self.assertEqual(self.env.drone.cell(), cell)
                self.assertEqual(info["drone_visited_map"][cell], 1)
                self.assertTrue(self.env.drone_observation_space.contains(info["drone_observation"]))

        for position in ([3.49, 3.49], [3.5, 3.5]):
            info = self.move_and_step(position)
            self.assertEqual(info["drone_reward_components"]["exploration"], 0)

    def test_history_resets_and_rewards_stay_zero_after_discovery(self):
        _, initial_info = self.env.reset(seed=42)
        initial_map = initial_info["drone_visited_map"]
        self.env.shared_comm[:] = [1, *self.env.target / 5]
        info = self.move_and_step([-3.25, -3.25])
        self.assertEqual(info["drone_reward"], 0)
        self.assertEqual(self.env.drone.visited_map[0, 0], 1)
        self.assertGreater(self.env.drone.visited_map.sum(), 1)

        _, info = self.env.reset(seed=42)
        np.testing.assert_array_equal(info["drone_visited_map"], initial_map)
        np.testing.assert_array_equal(info["drone_observation"][:3], [0, 0, 0])
        self.assertEqual(info["drone_visited_map"].sum(), 1)


if __name__ == "__main__":
    unittest.main()
