import unittest

from mcp.data.dataset.transforms import KorniaTransforms
from mcp.task.byol import BYOLTask
from mcp.task.compute import TaskCompute

MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)


class BYOLTest(unittest.TestCase):
    def test_initial_state_dict(self):
        transforms = KorniaTransforms(MEAN, STD, (10, 10), 2)
        compute = TaskCompute(transforms)
        byol = BYOLTask(128, compute, 64, 0.9)
        state_dict = byol.initial_state_dict
        self.assertIsNone(state_dict["momentum_encoder"])
        self.assertIsNone(state_dict["momentum_head_projection"])
