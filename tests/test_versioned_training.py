"""Regression checks for preserving checkpoints across training runs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import robot_experiment as experiment
from waste_collection_env import RobotMAEnv


class VersionedTrainingTest(unittest.TestCase):
    def test_resume_preserves_source_and_writes_a_new_run(self):
        for sorting, initial_two, resumed_two in (
            (False, False, False), (True, False, False),
            (True, True, True), (True, False, True),
        ):
            with self.subTest(sorting=sorting, initial_two=initial_two, resumed_two=resumed_two):
                self.check_resume(sorting, initial_two, resumed_two)

    def check_resume(self, sorting, initial_two, resumed_two):
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
                runner.train(steps=8, two_robots=initial_two)
                first = runner.last_run_directory
                source = first / 'final_model.zip'
                original_bytes = source.read_bytes()
                runner.train(steps=8, resume=True, two_robots=resumed_two)
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
            self.assertEqual(config['two_robots'], resumed_two)
            self.assertEqual(config['team_bonus'], 10 if sorting and resumed_two else 0)
            model = experiment.RolloutPPO.load(second / 'final_model.zip')
            self.assertEqual(model.num_timesteps, 8 * (2 + initial_two + resumed_two))
            self.assertEqual(model.n_envs, 2 if resumed_two else 1)
            self.assertEqual(model.observation_space.shape, (76 if sorting else 7,))
            self.assertEqual(model.best_output, str(second))
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

    def test_sorting_training_keeps_drone_policy_frozen(self):
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
                models = runner.train(steps=8, two_robots=True, drone_model=root / 'drone.zip')
            self.assertEqual(models['robot'].num_timesteps, 16)
            self.assertEqual(models['drone'].num_timesteps, 0)
            for key, value in models['drone'].policy.state_dict().items():
                self.assertTrue(experiment.torch.equal(value, original_weights[key]))


if __name__ == '__main__':
    unittest.main()
