"""Clearance rewards distinguish safe passage, proximity and physical contact."""

import unittest
import numpy as np

from waste_collection_env import WasteCollectionEnv


class RobotAvoidanceTests(unittest.TestCase):
    def test_clearance_and_collision(self):
        env = WasteCollectionEnv(sorting=True, two_robots=True)
        try:
            env.reset(seed=0)
            robot, other = env.robots.values()
            robot.go = 1
            def place(distance):
                qpos = env.init_qpos.copy()
                qpos[robot.qpos_indices] = [0, 0]
                qpos[other.qpos_indices] = [distance, 0]
                env.set_state(qpos, np.zeros(env.model.nv))
                return robot.compute_avoidance_reward()
            self.assertEqual(place(1.2), 0)
            near = place(0.6)
            self.assertLess(near, 0)
            self.assertGreater(near, -robot.robot_collision_penalty)
            self.assertLessEqual(place(0.3), -robot.robot_collision_penalty)
            robot.go = 0
            self.assertEqual(robot.compute_avoidance_reward(), 0)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
