import unittest
from shutil import rmtree

from mcp.result.logger import (
    ResultRecord,
    load_records,
    load_records_from_file,
    save_records,
)

OUTPUT = "/tmp/result_record_test"


RECORD = ResultRecord("task_name", 1.5, 55.2, "metric_name", 1)
OTHER_RECORD = ResultRecord("task_name", 0.2, 85.2, "metric_name", 1)

RECORD_TASK1 = ResultRecord("task_1", 1.5, 55.2, "metric_name", 1)
OTHER_RECORD_TASK1 = ResultRecord("task_1", 0.2, 85.2, "metric_name", 1)

RECORD_TASK2 = ResultRecord("task_2", 1.56, 65.2, "metric_name", 1)
OTHER_RECORD_TASK2 = ResultRecord("task_2", 0.299, 85.200, "metric_name", 1)


class ResultRecordsTest(unittest.TestCase):
    def setUp(self):
        rmtree(OUTPUT, ignore_errors=True)

    def test_shouldBeSerializable(self):
        with open(OUTPUT, "w") as file:
            save_records([RECORD], file)

        with open(OUTPUT, "r") as file:
            lines = file.readlines()

        record_actual = load_records(lines[0])[0]
        self.assertEqual(RECORD, record_actual)

    def test_shouldLoadFile(self):
        records = [
            [RECORD_TASK1, RECORD_TASK2],
            [OTHER_RECORD_TASK1, OTHER_RECORD_TASK2],
        ]
        with open(OUTPUT, "w") as file:
            for r in records:
                save_records(r, file)

        records_actual = load_records_from_file(OUTPUT)

        self.assertEqual(records, records_actual)
