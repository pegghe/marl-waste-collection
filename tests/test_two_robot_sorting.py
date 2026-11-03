"""MASB3 shares one policy between two specialists and keeps reports nonterminal."""

from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import Mock

import numpy as np
from masb3.utils import ma_evaluate

from waste_collection_env import RobotMAEnv


class TwoRobotSortingTest(unittest.TestCase):
    def setUp(self):
        self.ma_env = RobotMAEnv(sorting=True, two_robots=True)
        self.addCleanup(self.ma_env.close)
        self.ma_env.reset(seed=42)
        self.env = self.ma_env.env
        self.actions = {name: np.zeros(2) for name in self.env.robots}

    def place(self, material, reporter_near=True):
        env = self.env
        env.target_type = material
        env.target = np.array([1.2, 0.0])
        env.shared_comm.fill(0)
        env.team_reward_given = False
        for kind, body in enumerate(('target', 'paper')):
            env.model.body(body).pos[:2] = env.target if kind == material else [20, 20]
        qpos = env.data.qpos.copy()
        for role, robot in enumerate(env.robots.values()):
            robot.specialization = role
            robot.completed = False
            robot.detected_type = -1
            robot.go = 1
            robot.go_delay = 0
            env.model.body_quat[robot.visual_body_id] = [1, 0, 0, 0]
            near = (role != material) if reporter_near else (role == material)
            qpos[robot.qpos_indices] = [0, 0] if near else [-3, 1]
        env.set_state(qpos, np.zeros(env.model.nv))
        env.assign_robot()

    def test_resets_assign_complementary_roles_and_keep_both_spawns_distant(self):
        for seed in range(16):
            observations, info = self.ma_env.reset(seed=seed)
            robots = list(self.env.robots.values())
            self.assertEqual({r.specialization for r in robots}, {0, 1})
            self.assertEqual(info['robot_specializations'], {'robot': 'plastic', 'robot_2': 'cardboard'})
            for name, robot in self.env.robots.items():
                self.assertGreaterEqual(np.linalg.norm(robot.position - self.env.target), 3)
                self.assertEqual(observations[name].shape, (76,))
                self.assertTrue(self.ma_env.agents[name].observation_space.contains(observations[name]))
                self.assertEqual(info['robot_specializations'][name], robot.ray_categories[robot.specialization])
            self.assertGreaterEqual(np.linalg.norm(robots[0].position - robots[1].position), 1)
            nearest = min(self.env.robots, key=lambda name: np.linalg.norm(self.env.robots[name].position - self.env.target))
            self.assertEqual(info['active_robot'], nearest)
            for name in self.env.robots:
                self.assertEqual(observations[name][6], int(name == nearest))

    def test_masb3_batches_both_agents_into_one_shared_model(self):
        model = Mock()
        model.predict.side_effect = lambda obs: (np.zeros((len(obs), 2)), None)
        self.ma_env.set_agent_models({'robot': model})
        self.assertIs(self.ma_env.agents['robot'].model, self.ma_env.agents['robot_2'].model)
        self.env.max_steps = 2
        with redirect_stdout(StringIO()):
            per_agent, per_model = ma_evaluate(self.ma_env, {'robot': model}, total_episodes=1)
        self.assertEqual(model.predict.call_count, 2)
        for call in model.predict.call_args_list:
            self.assertEqual(call.args[0].shape, (2, 76))
        self.assertAlmostEqual(per_model['robot'], sum(per_agent.values()) / 2)

    def test_report_reaches_other_robot_and_episode_continues_until_collection(self):
        for material in (0, 1):
            with self.subTest(material=material):
                self.place(material)
                collector = list(self.env.robots.values())[material]
                reporter = list(self.env.robots.values())[1 - material]
                observations, _, terminated, truncated, info = self.ma_env.step_all(self.actions)
                self.assertTrue(info['reported'])
                self.assertEqual(info['team_reward'], 0)
                self.assertFalse(info['collected'])
                self.assertFalse(terminated or truncated)
                self.assertFalse(info['is_success'])
                self.assertTrue(reporter.completed)
                self.assertEqual(info['active_robot'], collector.name)
                self.assertEqual(observations[collector.name][6], 1)
                self.assertEqual(observations[reporter.name][6], 0)
                self.assertFalse(collector.completed)
                self.assertEqual(collector.detected_type, material)
                self.assertFalse(np.any(collector.ray_types == material))
                self.assertEqual(observations[collector.name][10 + material], 1)
                np.testing.assert_allclose(info['shared_comm'], [1, 1.2 / 5, 0, material + 1])
                # Collector can finish using the material communicated by its teammate.
                qpos = self.env.data.qpos.copy()
                qpos[reporter.qpos_indices] = [-3, -2]
                qpos[collector.qpos_indices] = [1.0, 0]
                self.env.set_state(qpos, np.zeros(self.env.model.nv))
                _, rewards, terminated, _, info = self.ma_env.step_all(self.actions)
                self.assertTrue(terminated)
                self.assertTrue(info['is_success'])
                self.assertTrue(info['collected'])
                self.assertFalse(info['wrong_target'])
                self.assertEqual(info['team_reward'], 10)
                self.assertEqual(info['individual_rewards'][reporter.name], 0)
                self.assertEqual(rewards[reporter.name], 10)
                for name in self.env.robots:
                    self.assertAlmostEqual(rewards[name], info['individual_rewards'][name] + 10)
                # Repeating a terminal step must not pay the team bonus again.
                _, _, _, _, repeated = self.ma_env.step_all(self.actions)
                self.assertEqual(repeated['team_reward'], 0)

    def test_collection_can_finish_without_a_report(self):
        for material in (0, 1):
            self.place(material, reporter_near=False)
            self.ma_env.step_all(self.actions)
            collector = list(self.env.robots.values())[material]
            qpos = self.env.data.qpos.copy()
            qpos[collector.qpos_indices] = [1.0, 0]
            self.env.set_state(qpos, np.zeros(self.env.model.nv))
            _, _, terminated, _, info = self.ma_env.step_all(self.actions)
            self.assertTrue(terminated)
            self.assertTrue(info['is_success'])
            self.assertTrue(info['collected'])
            self.assertFalse(info['reported'])

    def test_wrong_radius_is_checked_for_both_robots(self):
        for material in (0, 1):
            self.place(material)
            wrong = list(self.env.robots.values())[1 - material]
            qpos = self.env.data.qpos.copy()
            qpos[wrong.qpos_indices] = [1.0, 0]
            self.env.set_state(qpos, np.zeros(self.env.model.nv))
            _, rewards, terminated, _, info = self.ma_env.step_all(self.actions)
            self.assertTrue(terminated)
            self.assertTrue(info['wrong_target'])
            self.assertFalse(info['is_success'])
            self.assertEqual(info['team_reward'], 0)
            self.assertLess(rewards[wrong.name], -10)

    def test_invalid_joint_completion_does_not_receive_team_bonus(self):
        self.place(0)
        collector, wrong = self.env.robots.values()
        collector.detected_type = 0
        qpos = self.env.data.qpos.copy()
        qpos[collector.qpos_indices] = [1.0, 0]
        qpos[wrong.qpos_indices] = [1.4, 0]
        self.env.set_state(qpos, np.zeros(self.env.model.nv))
        self.env.assign_robot(material=0)
        _, _, terminated, _, info = self.ma_env.step_all(self.actions)
        self.assertTrue(terminated)
        self.assertTrue(info['collected'])
        self.assertFalse(info['is_success'])
        self.assertEqual(info['team_reward'], 0)

    def test_unassigned_robot_stays_still_and_has_no_individual_cost(self):
        self.place(0, reporter_near=False)
        inactive = self.env.robots['robot_2']
        position = inactive.position.copy()
        # Simulate residual velocity and a policy that keeps requesting motion.
        self.env.data.qvel[inactive.qvel_indices] = [1, -1]
        self.env.steps = 150
        _, _, terminated, _, info = self.ma_env.step_all({name: [1, 1] for name in self.env.robots})
        self.assertFalse(terminated)
        np.testing.assert_array_equal(inactive.position, position)
        np.testing.assert_array_equal(inactive.velocity, 0)
        np.testing.assert_array_equal(self.env.data.ctrl[inactive.actuator_ids], 0)
        self.assertEqual(inactive.go, 0)
        self.assertEqual(info['individual_rewards'][inactive.name], 0)
        self.assertEqual(self.env.active_robot, 'robot')

    def test_assignment_does_not_switch_when_distances_change_and_ties_are_stable(self):
        self.place(0, reporter_near=False)
        qpos = self.env.data.qpos.copy()
        qpos[self.env.robots['robot'].qpos_indices] = [-2, 0]
        qpos[self.env.robots['robot_2'].qpos_indices] = [0, 0]
        self.env.set_state(qpos, np.zeros(self.env.model.nv))
        self.ma_env.step_all(self.actions)
        self.assertEqual(self.env.active_robot, 'robot')
        # A new assignment uses the current distances, with robot winning an exact tie.
        qpos[self.env.robots['robot'].qpos_indices] = [0, -1]
        qpos[self.env.robots['robot_2'].qpos_indices] = [0, 1]
        self.env.set_state(qpos, np.zeros(self.env.model.nv))
        self.env.assign_robot()
        self.assertEqual(self.env.active_robot, 'robot')


if __name__ == '__main__':
    unittest.main()
