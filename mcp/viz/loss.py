import os

from mcp.result.experiment import EpochResult, ExperimentResult
from mcp.viz.line_plot import line_plot


def plot_loss(output_dir: str, results: ExperimentResult):
    losses_train = results.metric("train", EpochResult.losses, reduce_task=None)
    task_names_train = results.task_names("train")

    losses_eval = results.metric("eval", EpochResult.losses, reduce_task=None)
    task_names_eval = results.task_names("eval")

    fig = line_plot(
        task_names_train, task_names_eval, losses_train, losses_eval, "Loss"
    )

    file_name = os.path.join(output_dir, "losses.png")
    fig.savefig(file_name)
