from typing import NewType

from mcp.task.base import Task

TasksTrain = NewType("TasksTrain", list)
TasksValid = NewType("TasksValid", list)
TaskTest = NewType("TaskTest", Task)
