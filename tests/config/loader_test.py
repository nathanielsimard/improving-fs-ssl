import unittest

from mcp.config import loader


class ConfigTest(unittest.TestCase):
    def test_parseConfig_shouldMergeComplexDict(self):
        default = {
            "key1": "value1",
            "key2": {
                "key3": "value2",
                "key4": "value3",
                "key5": {"key6": "value4", "key7": "value5"},
            },
        }
        config = {
            "key2": {"key3": "override-value2", "key5": {"key7": "override-value5"}}
        }
        expected = {
            "key1": "value1",
            "key2": {
                "key3": "override-value2",
                "key4": "value3",
                "key5": {"key6": "value4", "key7": "override-value5"},
            },
        }

        actual = loader.merge(config, default=default)

        self.assertEqual(expected, actual)

    def test_parseConfig_shouldWorkWithList(self):
        default = {
            "key1": [{"key2": "value1"}, {"key2": "value2"}],
            "key3": {"key4": ["value3", "value4", "value5"]},
            "key4": ["value6"],
        }
        config = {
            "key1": [{"key2": "overrided-value2"}],
            "key3": {"key4": ["new-list-item1", "new-list-item2"]},
        }
        expected = {
            "key1": [{"key2": "overrided-value2"}],
            "key3": {"key4": ["new-list-item1", "new-list-item2"]},
            "key4": ["value6"],
        }

        actual = loader.merge(config, default=default)

        self.assertEqual(expected, actual)

    def test_parseConfig_shouldRaiseWhenUnknownKey(self):
        default = {
            "key1": "value1",
            "key2": {
                "key3": "value2",
                "key4": "value3",
                "key5": {"key6": "value4", "key7": "value5"},
            },
        }
        config = {
            "key2": {"key3": "override-value2", "key5": {"unknown_key": "somevalue"}}
        }

        self.assertRaises(ValueError, lambda: loader.merge(config, default=default))
