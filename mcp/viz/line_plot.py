from typing import List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


def line_plot(
    task_names_train: List[str],
    task_names_eval: List[str],
    values_train: np.ndarray,
    values_eval: np.ndarray,
    y_label: str,
    x_label: str = "Epoch",
    bbox_to_anchor=(0.90, 0.88),
    y_int: bool = False,
    x_int: bool = True,
) -> plt.Figure:

    fig = plt.figure()
    ax = fig.subplots()
    for i, name in enumerate(task_names_train):
        x = list(range(len(values_train)))
        ax.plot(
            x, values_train[:, i], label=f"Train - {name}",
        )

    for i, name in enumerate(task_names_eval):
        x = list(range(len(values_eval)))
        ax.plot(
            x, values_eval[:, i], label=f"Valid - {name}", linestyle=":",
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if x_int:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if y_int:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.legend(bbox_to_anchor=bbox_to_anchor)
    return fig
