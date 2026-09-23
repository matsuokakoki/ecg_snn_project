import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation_protocol import (  # noqa: E402
    split_training_records,
    subject_id,
    validate_subject_disjoint,
)


class EvaluationProtocolTests(unittest.TestCase):
    def test_known_shared_subject(self):
        self.assertEqual(subject_id("201"), subject_id("202"))
        with self.assertRaises(ValueError):
            validate_subject_disjoint(["201"], ["202"])

    def test_split_keeps_shared_subject_together(self):
        records = ["100", "101", "102", "103", "201", "202"]
        train, val = split_training_records(records, seed=42)
        self.assertEqual(set(train) | set(val), set(records))
        validate_subject_disjoint(train, val)
        self.assertEqual("201" in train, "202" in train)


if __name__ == "__main__":
    unittest.main()
