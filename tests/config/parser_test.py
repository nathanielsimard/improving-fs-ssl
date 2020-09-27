import unittest

from mcp.config.loader import to_dict
from mcp.config.parser import DEFAULT_CONFIG, parse


class ExperimentConfigTest(unittest.TestCase):
    def test_whenEmptyConfig_shouldBeDefaultConfig(self):
        config_experiment = parse([], default=DEFAULT_CONFIG)
        config = to_dict(config_experiment)

        self.assertEqual(config, DEFAULT_CONFIG)
