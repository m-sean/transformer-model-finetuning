import ast
import json
from typing import Dict, Iterable, Tuple

import pandas as pd
from torch.utils.data import Dataset


def _collect_categorical(records, label_mapping, x_field, y_field):
    x = []
    y = []
    for rec in records:
        x.append(rec[x_field])
        label = rec[y_field]
        if label_mapping is not None:
            label = label_mapping[label]
        if isinstance(label, str):
            raise ValueError(
                "Labels/targets must be represented as float or int values"
            )
        y.append(label)
    return x, y


def _collect_multilabel(records, label_mapping, x_field, y_field):
    x = []
    y = []
    tmp_vec = [0 for _ in range(len(label_mapping))]
    for rec in records:
        x.append(rec[x_field])
        rec_labels = rec[y_field]
        if isinstance(rec_labels, str):
            if rec_labels.startswith("["):
                rec_labels = ast.literal_eval(rec_labels)
            elif "," in rec_labels:
                rec_labels = [label.strip() for label in rec_labels.split(",")]
            else:
                raise ValueError(f"Unable to parse record labels: {rec_labels}")
        rec_labels_vec = tmp_vec.copy()
        for label in rec_labels:
            label_idx = label_mapping[label]
            rec_labels_vec[label_idx] = 1
        y.append(rec_labels_vec)
    return x, y


def _iter_tokenized_text_df(text_df: pd.DataFrame, text_field: str, label_field: str):
    x_tmp = []
    y_tmp = []
    for row in text_df.itertuples():
        text = getattr(row, text_field) or None
        if pd.isna(text) and x_tmp:
            yield {text_field: x_tmp, label_field: y_tmp}
            x_tmp = []
            y_tmp = []
            continue
        x_tmp.append(text)
        y_tmp.append(getattr(row, label_field))
    if x_tmp:
        yield {text_field: x_tmp, label_field: y_tmp}


class BaseTextDataset(Dataset):

    def __init__(self) -> None:
        self.x = []
        self.y = []
        super(BaseTextDataset, self).__init__()

    def __getitem__(self, idx) -> Tuple[str, int]:
        return self.x[idx], self.y[idx]

    def __len__(self) -> int:
        return len(self.y)

    @classmethod
    def from_jsonl(
        cls,
        file: str,
        label_mapping: Dict[str, int] = None,
        x_field: str = "content",
        y_field: str = "label",
    ):
        with open(file, "r", encoding="utf-8") as src:
            records = list(json.loads(line) for line in src)
            return cls(records, label_mapping, x_field, y_field)

    @classmethod
    def from_csv(
        cls,
        file: str,
        label_mapping: Dict[str, int] = None,
        x_field: str = "content",
        y_field: str = "label",
    ):
        with open(file, "r") as src:
            records = pd.read_csv(src).to_dict(orient="records")
            return cls(records, label_mapping, x_field, y_field)


class TextDataset(BaseTextDataset):
    def __init__(
        self,
        records: Iterable[dict],
        label_mapping: Dict[str, int] = None,
        x_field: str = "content",
        y_field: str = "label",
    ) -> None:
        super(TextDataset, self).__init__()
        self.x, self.y = _collect_categorical(records, label_mapping, x_field, y_field)


class TokenizedTextDataset(BaseTextDataset):
    def __init__(
        self,
        records: Iterable[dict],
        label_mapping: Dict[str, int] = None,
        x_field: str = "content",
        y_field: str = "label",
    ) -> None:
        super(TokenizedTextDataset, self).__init__()
        self.x = []
        self.y = []
        for rec in records:
            self.x.append(rec[x_field])
            labels = rec[y_field]
            self.y.append([label_mapping[lbl] for lbl in labels])

    @classmethod
    def from_csv(
        cls,
        file: str,
        label_mapping: Dict[str, int] = None,
        x_field: str = "tokens",
        y_field: str = "tags",
    ):
        with open(file, "r") as src:
            records = pd.read_csv(src)
            records_as_dicts_iter = _iter_tokenized_text_df(records, x_field, y_field)
            return cls(records_as_dicts_iter, label_mapping, x_field, y_field)


class TextMultiLabelDataset(BaseTextDataset):
    def __init__(
        self,
        records: Iterable[dict],
        label_mapping: Dict[str, int] = None,
        x_field: str = "content",
        y_field: str = "label",
    ) -> None:
        super(TextMultiLabelDataset, self).__init__()
        self.x, self.y = _collect_multilabel(records, label_mapping, x_field, y_field)
