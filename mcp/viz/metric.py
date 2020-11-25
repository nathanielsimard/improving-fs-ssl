import os

import numpy as np

from mcp.result.experiment import EpochResult, ExperimentResult
from mcp.viz.line_plot import line_plot


def plot_metric(output_dir: str, results: ExperimentResult):
    metric_train = results.metric("train", EpochResult.metric, reduce_task=None)
    task_names_train = results.task_names("train")

    metric_eval = results.metric("eval", EpochResult.metric, reduce_task=None)
    task_names_eval = results.task_names("eval")

    metric_names_train = results.metric_names("train")
    metric_names_eval = results.metric_names("eval")

    for i, (task, metric) in enumerate(zip(task_names_train, metric_names_train)):
        fig = line_plot([task], [], metric_train[:, i : i + 1], np.array([]), metric)
        file_name = os.path.join(output_dir, f"metric-{task}-{metric}-train.png")
        fig.savefig(file_name)

    for i, (task, metric) in enumerate(zip(task_names_eval, metric_names_eval)):
        fig = line_plot([], [task], np.array([]), metric_eval[:, i : i + 1], metric)
        file_name = os.path.join(output_dir, f"metric-{task}-{metric}-eval.png")
        fig.savefig(file_name)
