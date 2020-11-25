import os

from mcp.result.experiment import ExperimentResult
from mcp.viz.loss import plot_loss
from mcp.viz.metric import plot_metric


class Vizualization(object):
    def __init__(self, results: ExperimentResult):
        self.results = results

    def plot(self):
        output_dir = os.path.join(self.results.output_dir, "viz")
        os.makedirs(output_dir, exist_ok=True)

        plot_loss(output_dir, self.results)
        plot_metric(output_dir, self.results)
