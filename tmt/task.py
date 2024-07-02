import enum
import torch
from tmt.data import TextDataset, TextMultiLabelDataset, TokenizedTextDataset
from tmt.util import (
    binary_task_output,
    multiclass_task_output,
    multilabel_task_output,
    regression_task_output,
)

_TASK_MAPPING = {
    "BINARY":{
        "ds_class": TextDataset,
        "dtype": torch.float,
        "metric": "mcc",
        "processor": binary_task_output,
    },
    "MULTICLASS": {
        "ds_class": TextDataset,
        "dtype": torch.long,
        "metric": "rk",
        "processor": multiclass_task_output,
    },
    "MULTICLASS_SEQUENCE": {
        "ds_class": TokenizedTextDataset,
        "dtype": torch.long,
        "metric": "rk",
        "processor": multiclass_task_output,
    },
    "MULTILABEL": {
        "ds_class": TextMultiLabelDataset,
        "dtype": torch.float,
        "metric": "Hmcc",
        "processor": multilabel_task_output,
    },
    "REGRESSION": {
        "ds_class": TextDataset,
        "dtype": torch.float,
        "metric": "r2",
        "processor": regression_task_output,
    },
}


class Task(enum.Enum):
    BINARY = 0
    MULTICLASS = 1
    MULTICLASS_SEQUENCE = 2
    MULTILABEL = 3
    REGRESSION = 4

    def to_str(self) -> str:
        return self.name.lower()
    
    def metric(self) -> str:
        return _TASK_MAPPING[self.name]["metric"]

    @classmethod
    def from_str(cls, task: str):
        return getattr(cls, task.upper())

    def output_processor(self):
        return _TASK_MAPPING[self.name]["processor"]

    def label_dtype(self):
        return _TASK_MAPPING[self.name]["dtype"]

    def dataset_class(self) -> TextDataset | TokenizedTextDataset:
        return _TASK_MAPPING[self.name]["ds_class"]

    def is_classification(self):
        return self != Task.REGRESSION
