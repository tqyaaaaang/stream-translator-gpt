import unittest

import tyro

from stream_translator_gpt.utils.logging_utils import normalize_verbosity_args
from stream_translator_gpt.main import Config


class CliVerbosityTest(unittest.TestCase):

    def test_supported_verbosity_forms(self):
        cases = [
            (['-v'], 1),
            (['-vv'], 2),
            (['-vvv'], 3),
            (['--verbose', '2'], 2),
            (['--verbose=3'], 3),
        ]
        for extra_args, expected in cases:
            with self.subTest(extra_args=extra_args):
                config = tyro.cli(Config, args=['input'] + normalize_verbosity_args(extra_args))
                self.assertEqual(config.verbose, expected)

    def test_gpt_verbosity_option_remains_independent(self):
        config = tyro.cli(Config, args=['input', '--verbosity', 'high'])

        self.assertEqual(config.verbose, 0)
        self.assertEqual(config.verbosity, 'high')


if __name__ == '__main__':
    unittest.main()
