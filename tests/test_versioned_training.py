"""Regression checks for preserving checkpoints across training runs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import robot_experiment as experiment
from waste_collection_env import RobotMAEnv


class VersionedTrainingTest(unittest.TestCase):
    def test_resume_preserves_source_and_writes_a_new_run(self):
        for sorting, initial_count, resumed_count in (
            (False, 1, 1), (True, 1, 1), (True, 2, 2), (True, 1, 2),
            (True, 2, 3), (True, 3, 3), (True, 3, 6),
        ):
            with self.subTest(sorting=sorting, initial_count=initial_count, resumed_count=resumed_count):
                self.check_resume(sorting, initial_count, resumed_count)

    def check_resume(self, sorting, initial_count, resumed_count):
        original_train = experiment.ma_train
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = experiment.RobotExperiment(str(root / 'final_model'), sorting=sorting)

            def short_training(**kwargs):
                for _, params in kwargs['model_algo_map'].values():
                    params.update(n_steps=8, batch_size=8, n_epochs=1,
                                  tensorboard_log=str(root / 'tensorboard'), verbose=0)
                return original_train(**kwargs)

            with patch.object(experiment, 'ma_train', side_effect=short_training):
                runner.train(steps=8, two_robots=initial_count == 2, three_robots=initial_count == 3)
                first = runner.last_run_directory
                source = first / 'final_model.zip'
                original_bytes = source.read_bytes()
                runner.train(steps=8, resume=True, two_robots=resumed_count == 2,
                             three_robots=resumed_count == 3,
                             robots_per_team=(2, 2, 2) if resumed_count == 6 else None,
                             waste_per_type=(2, 2, 2) if resumed_count == 6 else None)
                second = runner.last_run_directory

            self.assertEqual(first.name, 'run_001')
            self.assertEqual(second.name, 'run_002')
            self.assertNotEqual(first, second)
            self.assertEqual(source.read_bytes(), original_bytes)
            self.assertFalse((root / 'final_model.zip').exists())
            self.assertTrue((second / 'final_model_vecnormalize.pkl').is_file())
            self.assertEqual(runner.resolve_model('robot'), second / 'final_model.zip')
            self.assertEqual(runner.resolve_model('robot', first), source)
            config = json.loads((second / 'config.json').read_text())
            self.assertEqual(config['resume_from'], str(source.resolve()))
            self.assertEqual(config['sorting'], sorting)
            self.assertEqual(config['two_robots'], resumed_count == 2)
            self.assertEqual(config['three_robots'], resumed_count == 3)
            self.assertEqual(config['robot_count'], resumed_count)
            self.assertEqual(config['materials'], ['plastic', 'cardboard', 'residual'][:
                3 if resumed_count >= 3 else (2 if sorting else 1)
            ])
            self.assertEqual(config['team_bonus'], 10 if sorting and resumed_count > 1 else 0)
            model = experiment.RolloutPPO.load(second / 'final_model.zip')
            self.assertEqual(model.num_timesteps, 8 * (initial_count + resumed_count))
            self.assertEqual(model.n_envs, resumed_count)
            self.assertEqual(model.observation_space.shape, (76 if sorting else 7,))
            self.assertEqual(model.best_output, str(second))
            if resumed_count == 6:
                self.assertEqual(config['robots_per_team'], [2, 2, 2])
                self.assertEqual(config['waste_per_type'], [2, 2, 2])
                self.assertEqual(config['assignment'], 'sequential_nearest')
            # Best must belong to the selected run; never silently use another run.
            with self.assertRaises(FileNotFoundError):
                runner.resolve_model('robot', second, best=True)
            (first / 'best_model.zip').write_bytes(original_bytes)
            self.assertEqual(runner.resolve_model('robot', first, best=True), first / 'best_model.zip')
            # An unfinished newer run must not become the default.
            (root / 'runs' / 'zz-incomplete').mkdir()
            self.assertEqual(runner.resolve_model('robot'), second / 'final_model.zip')

    def test_legacy_checkpoint_remains_selectable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'final_model.zip').write_bytes(b'legacy')
            runner = experiment.RobotExperiment(str(root / 'final_model'))
            self.assertEqual(runner.resolve_model('robot'), root / 'final_model.zip')
            with self.assertRaises(FileNotFoundError):
                runner.resolve_model('robot', root / 'missing.zip')

    def test_configurable_training_records_seed_and_world_step_budget(self):
        original_train = experiment.ma_train
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = experiment.RobotExperiment(str(root / 'final_model'), sorting=True)

            def short_training(**kwargs):
                self.assertEqual(kwargs['total_timesteps_per_model'], 24)
                for _, params in kwargs['model_algo_map'].values():
                    params.update(n_steps=8, batch_size=8, n_epochs=1,
                                  tensorboard_log=str(root / 'tensorboard'), verbose=0)
                return original_train(**kwargs)

            with patch.object(experiment, 'ma_train', side_effect=short_training):
                runner.train(steps=8, worldsteps=True, seed=17,
                             robots_per_team=(1, 1, 1), waste_per_type=(2, 2, 2))
            config = json.loads((runner.last_run_directory / 'config.json').read_text())
            self.assertEqual(config['seed'], 17)
            self.assertEqual(config['step_unit'], 'world')
            self.assertEqual(config['requested_steps'], 8)
            self.assertEqual(config['requested_learning_steps'], 24)
            self.assertEqual(config['learning_agents'], 3)
            model = experiment.RolloutPPO.load(runner.last_run_directory / 'final_model.zip')
            self.assertEqual(model.num_timesteps, 24)
            self.assertEqual(model.seed, 17)

    def test_evaluation_applies_requested_seed_after_loading_the_checkpoint(self):
        original_evaluate = experiment.ma_evaluate
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_env = experiment.MultiAgentSharedVecEnv(
                RobotMAEnv(sorting=True, three_robots=True), ['robot', 'robot_2', 'robot_3'],
            )
            try:
                model = experiment.get_algo('sb3', 'PPO')(
                    'MlpPolicy', source_env, n_steps=8, batch_size=8, seed=17, device='cpu',
                )
                model.save(root / 'source.zip')
            finally:
                source_env.close()

            def short_evaluation(env, models, **kwargs):
                self.assertEqual(experiment.torch.initial_seed(), 29)
                self.assertEqual(np.random.get_state()[1][0], 29)
                env.env.max_steps = 2
                return original_evaluate(env, models, **kwargs)

            runner = experiment.RobotExperiment(sorting=True)
            with patch.object(experiment, 'ma_evaluate', side_effect=short_evaluation):
                runner.evaluate(model=root / 'source.zip', episodes=1, seed=29,
                                robots_per_team=(1, 1, 1), waste_per_type=(2, 2, 2))

    def test_sorting_training_keeps_drone_policy_frozen(self):
        for robot_count in (2, 3):
            with self.subTest(robot_count=robot_count):
                self.check_frozen_drone(robot_count)

    def check_frozen_drone(self, robot_count):
        original_train = experiment.ma_train
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            drone_env = experiment.MultiAgentSharedVecEnv(RobotMAEnv(drone_training=True), ['drone'])
            try:
                drone = experiment.get_algo('sb3', 'PPO')(
                    'MlpPolicy', drone_env, n_steps=8, batch_size=8, device='cpu', seed=0,
                )
                drone.save(root / 'drone.zip')
                original_weights = {key: value.clone() for key, value in drone.policy.state_dict().items()}
            finally:
                drone_env.close()

            def short_training(**kwargs):
                self.assertEqual(kwargs['models_to_train'], ['robot'])
                for _, params in kwargs['model_algo_map'].values():
                    params.update(n_steps=8, batch_size=8, n_epochs=1,
                                  tensorboard_log=str(root / 'tensorboard'), verbose=0)
                return original_train(**kwargs)

            runner = experiment.RobotExperiment(str(root / 'robot' / 'final_model'), sorting=True)
            with patch.object(experiment, 'ma_train', side_effect=short_training):
                models = runner.train(steps=8, two_robots=robot_count == 2,
                                      three_robots=robot_count == 3, drone_model=root / 'drone.zip')
            self.assertEqual(models['robot'].num_timesteps, 8 * robot_count)
            self.assertEqual(models['drone'].num_timesteps, 0)
            for key, value in models['drone'].policy.state_dict().items():
                self.assertTrue(experiment.torch.equal(value, original_weights[key]))


if __name__ == '__main__':
    unittest.main()
