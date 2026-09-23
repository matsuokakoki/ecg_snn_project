"""Split helpers for the corrected, as-yet-unrun evaluation protocol.

MIT-BIH records 201 and 202 belong to one subject. A record-level split that
separates them is not a subject-level split. See the PhysioNet directory linked
from README.md.
"""

import random


def subject_id(record_id: str) -> str:
    record_id = str(record_id)
    return "201-202" if record_id in {"201", "202"} else record_id


def validate_subject_disjoint(*record_sets) -> None:
    subjects = [set(map(subject_id, records)) for records in record_sets]
    for left in range(len(subjects)):
        for right in range(left + 1, len(subjects)):
            overlap = subjects[left] & subjects[right]
            if overlap:
                raise ValueError(f"Subjects overlap across partitions: {sorted(overlap)}")


def split_training_records(records, validation_fraction=0.2, seed=42):
    """Return a deterministic subject-disjoint train/validation record split."""
    groups = {}
    for record in records:
        groups.setdefault(subject_id(record), []).append(str(record))
    if len(groups) < 3:
        raise ValueError("At least three subjects are needed for a validation split")
    subjects = sorted(groups)
    random.Random(seed).shuffle(subjects)
    count = max(1, round(len(subjects) * validation_fraction))
    if count >= len(subjects):
        raise ValueError("Validation would consume every training subject")
    validation = {record for subject in subjects[:count] for record in groups[subject]}
    train = [str(record) for record in records if str(record) not in validation]
    val = [str(record) for record in records if str(record) in validation]
    validate_subject_disjoint(train, val)
    return train, val
