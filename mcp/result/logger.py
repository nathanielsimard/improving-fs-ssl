import json
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, TextIO

from mcp.task.base import TaskOutput
from mcp.utils.logging import create_logger

logger = create_logger(__name__)


@dataclass
class ResultRecord:
    name: str
    loss: float
    metric: float
    metric_name: str
    time: float

    def serialize(self) -> str:
        return json.dumps(
            {
                "name": self.name,
                "loss": str(self.loss),
                "metric": str(self.metric),
                "metric_name": self.metric_name,
                "time": self.time,
            }
        )


def load_records_from_file(file_path: str) -> List[List[ResultRecord]]:
    with open(file_path, "r") as file:
        return [load_records(line) for line in file]


def load_records(line: str) -> List[ResultRecord]:
    """Load all records for an iteration.

    The number of records is determined by the number of tasks.
    """
    objs = json.loads(line)
    return [
        ResultRecord(
            name=obj["name"],
            loss=float(obj["loss"]),
            metric=float(obj["metric"]),
            metric_name=obj["metric_name"],
            time=obj["time"],
        )
        for obj in objs
    ]


def save_records(records: List[ResultRecord], file: TextIO):
    records_str = [r.serialize() for r in records]
    content = "[" + ",".join(records_str) + "]\n"
    file.write(content)


class ResultLogger(object):
    def __init__(self, str_format: str, output: str):
        self.str_format = str_format
        self.output = output

        self._file: Optional[TextIO] = None
        self._epoch = 0
        self._epochs = 0

    def log(
        self,
        outputs: List[TaskOutput],
        task_names: List[str],
        batch_idx: int,
        num_batches: int,
    ):
        def format(value: float) -> str:
            if value < 0.001 and value != 0:
                return f"{value:.3e}"
            else:
                return f"{value:.3f}"

        info = [
            f"{n}: loss={format(o.loss.item())} {o.metric_name}={format(o.metric)}"
            for n, o in zip(task_names, outputs)
        ]

        logger.info(
            f"{self.str_format} - Epoch {self._epoch}/{self._epochs} Batch {batch_idx}/{num_batches} {' | '.join(info)}"
        )

        if self._file is None:
            raise Exception("Training logger not initialized with a specific epoch.")

        records = [
            ResultRecord(
                name=name,
                loss=output.loss.item(),
                metric=output.metric,
                metric_name=output.metric_name,
                time=output.time,
            )
            for name, output in zip(task_names, outputs)
        ]
        save_records(records, self._file)

    def epoch(self, epoch: int, epochs: int):
        if self._file is not None:
            self._file.close()
            self._file = None

        logger_copy = deepcopy(self)
        logger_copy.output = self.output + f"-{epoch}"
        logger_copy._file = open(logger_copy.output, "w")
        logger_copy._epoch = epoch
        logger_copy._epochs = epochs
        return logger_copy

    def __del__(self):
        if self._file is not None:
            self._file.close()
            self._file = None
