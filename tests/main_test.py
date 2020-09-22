import unittest

from mcp import main


class MainTest(unittest.TestCase):
    def test_main(self):
        self.assertTrue(main.main)
