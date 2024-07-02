import os
from functools import partial
from tqdm import tqdm
from typing import Dict, List, Optional

import torch
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy

from tmt.loss import LOSS_FUNCTIONS
from tmt.model import TransformerModel, TransformerSequenceModel
from tmt.optimizer import OptimizerConfig
from tmt.task import Task
from tmt.util import apply_mask, pad_sequence

def _batch_collate_fn(
    batch: List[str],
    tokenizer: PreTrainedTokenizerFast,
    device: torch.device,
    task: Task,
):
    sequence_task = task == Task.MULTICLASS_SEQUENCE
    x, y = zip(*batch)
    encodings = tokenizer(
        x,
        padding=True,
        truncation=True,
        return_attention_mask=True,
        is_split_into_words=sequence_task,
        add_special_tokens=False,
        return_tensors="pt",
    )
    x = encodings["input_ids"].to(device)
    mask = encodings["attention_mask"].to(device)
    if sequence_task:
        y = [pad_sequence(y_seq, x.shape[1], -1) for y_seq in y]
    y = torch.tensor(y, dtype=task.label_dtype(), device=device)
    return x, mask, y

class TransformerModelHandler:
    def __init__(
        self,
        model_name: str,
        task: Task,
        label_mapping: Dict[str, int] = None,
        batch_size: int = 32,
        activation_fn: str = None,
    ) -> None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, add_prefix_space=task == Task.MULTICLASS_SEQUENCE
        )
        max_input_len = tokenizer.model_max_length
        try:
            tokenizer.set_truncation_and_padding(
                PaddingStrategy.LONGEST,
                TruncationStrategy.LONGEST_FIRST,
                max_length=max_input_len,
                stride=0,
                pad_to_multiple_of=None,
            )
        except Exception as err:
            print(f"Error setting max seq len: {err}. Defaulting to 512.")
            max_input_len = 512
            tokenizer.model_max_length = max_input_len
            tokenizer.set_truncation_and_padding(
                PaddingStrategy.LONGEST,
                TruncationStrategy.LONGEST_FIRST,
                max_length=max_input_len,
                stride=0,
                pad_to_multiple_of=None,
            )
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_input = max_input_len
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task = task
        collate_fn = partial(
            _batch_collate_fn,
            tokenizer=self.tokenizer,
            device=self.device,
            task=self.task,
        )
        self.label_mapping = label_mapping
        self.collate_fn = collate_fn
        self.optimizer = None
        self.norm_clip = None
        if activation_fn and self.task != Task.REGRESSION:
            activation_fn = None
        self.activation_fn = activation_fn

    def model_output_dim(self) -> int:
        if self.task in {Task.BINARY, Task.REGRESSION}:
            return 1
        return len(self.label_mapping)

    def get_idx_to_label(self) -> Dict[int, str]:
        return {idx: label for label, idx in self.tag_to_idx.items()}

    def get_data_loader(
        self, data: str, train: bool, x_field: str, y_field: str
    ) -> DataLoader:
        ds_class = self.task.dataset_class()
        if data.endswith("jsonl"):
            dataset = ds_class.from_jsonl(data, self.label_mapping, x_field, y_field)
        elif data.endswith("csv"):
            dataset = ds_class.from_csv(data, self.label_mapping, x_field, y_field)
        else:
            raise ValueError("Dataset must be in csv or jsonl format")
        return DataLoader(
            dataset, self.batch_size, shuffle=train, collate_fn=self.collate_fn
        )

    def empty_loader(self):
        return DataLoader([], self.batch_size, collate_fn=self.collate_fn)

    def process_output(
        self,
        loss_fn: torch.nn.Module,
        outputs: torch.Tensor,
        mask: torch.Tensor,
        targets: torch.Tensor,
    ):
        if self.task == Task.MULTICLASS_SEQUENCE:
            outputs, targets = apply_mask(outputs, targets, mask, outputs.shape[-1])
        processor = self.task.output_processor()
        if self.activation_fn:
            processor = partial(processor, activation=self.activation_fn)
        loss, metrics = processor(outputs, targets, loss_fn)
        return loss, metrics

    def train(
        self,
        train: DataLoader,
        val: DataLoader,
        epochs: int,
        save: str,
        optimizer: OptimizerConfig,
        loss: torch.nn.Module,
        norm_clip: Optional[float] = None,
        scaler: Optional[GradScaler] = None,
        save_top: int = 1,
        test: DataLoader | None = None,
        dropout: float = 0.0,
    ) -> TransformerModel:
        self.norm_clip = norm_clip
        model_cls = (
            TransformerModel
            if self.task != Task.MULTICLASS_SEQUENCE
            else TransformerSequenceModel
        )
        metric_key = self.task.metric()
        model = model_cls(
            model_name=self.model_name,
            input_dim=self.max_input,
            output_dim=self.model_output_dim(),
            task=self.task.to_str(),
            dropout=dropout,
        ).to(device=self.device)
        model.train()
        self.optimizer = optimizer.get_optimizer(model.trainable_params())
        batch_ct = len(train)
        loss_fn = LOSS_FUNCTIONS[loss]()
        save_history = []
        best_result = float("-inf")
        ignore_metrics = save_top == epochs
        for ep in range(1, epochs + 1):
            with tqdm(total=batch_ct, unit="batch") as pbar:
                pbar.set_description(f"Epoch {ep}")
                if scaler:
                    t_metrics = self._train_step_amp(model, train, loss_fn, pbar, scaler)
                else:
                    t_metrics = self._train_step(model, train, loss_fn, pbar)
                v_metrics = self._val_step(model, val, loss_fn, pbar)
                t_report = " ".join(
                    [f"t_{metric}={value:.3f}" for metric, value in t_metrics.items()]
                )
                v_report = " ".join(
                    [
                        f"v_{metric}={value if value is not None else float('nan'):.3f}"
                        for metric, value in v_metrics.items()
                    ]
                )
                pbar.set_postfix_str(f"{t_report} {v_report}")

            if ignore_metrics:
                model.save(save, f"e{ep}")
                continue

            if (epoch_metric := v_metrics.get(metric_key)) and epoch_metric > best_result:
                best_result = epoch_metric
                model.save(save, f"e{ep}")
                save_history.append(ep)
                if len(save_history) > save_top:
                    ckpt = save_history.pop(0)
                    os.remove(f"{save}/model_e{ckpt}.pt")
        if test:
            model = model_cls.load(save, f"e{save_history[-1]}")
            batch_ct = len(test)
            with tqdm(total=batch_ct, unit="batch") as pbar:
                eval_metrics = self._val_step(
                    model, test, loss_fn, pbar, expanded_metrics=True
                )
            for metric, value in eval_metrics.items():
                print(f"{metric.upper()}: {value or float('nan'):.4f}")
        return model

    def _train_step(
        self,
        model: TransformerModel,
        train: DataLoader,
        loss_fn: torch.nn.Module,
        pbar: tqdm,
    ) -> Dict[str, float]:
        sum_metrics = {
            "loss": 0.0,
        }
        model.train()
        for i, (x, mask, targets) in enumerate(train):
            model.zero_grad()
            outputs = model(input_ids=x, attention_mask=mask)
            batch_loss, batch_metrics = self.process_output(
                loss_fn, outputs, mask, targets
            )
            batch_loss.backward()
            if self.norm_clip:
                clip_grad_norm_(model.parameters(), self.norm_clip)
            self.optimizer.step()

            for metric, value in batch_metrics.items():
                sum_metrics[metric] = sum_metrics.get(metric, 0.0) + (value or 0.0)
            n = i + 1
            sum_metrics["loss"] += batch_loss
            results = {metric: value / n for metric, value in sum_metrics.items()}
            pbar.update()
            pbar.set_postfix_str(" ".join(f"{met}={val:.3f}" for met, val in results.items()))
        return results
    
    def _train_step_amp(
        self,
        model: TransformerModel,
        train: DataLoader,
        loss_fn: torch.nn.Module,
        pbar: tqdm,
        scaler: GradScaler,
    ) -> Dict[str, float]:
        sum_metrics = {
            "loss": 0.0,
        }
        model.train()
        for i, (x, mask, targets) in enumerate(train):
            model.zero_grad()
            with torch.autocast(device_type=self.device.type):
                outputs = model(input_ids=x, attention_mask=mask)
                batch_loss, batch_metrics = self.process_output(
                    loss_fn, outputs, mask, targets
                )
            scaler.scale(batch_loss).backward()
            if self.norm_clip:
                scaler.unscale_(self.optimizer)
                clip_grad_norm_(model.parameters(), self.norm_clip)
            scaler.step(self.optimizer)
            scaler.update()

            for metric, value in batch_metrics.items():
                sum_metrics[metric] = sum_metrics.get(metric, 0.0) + (value or 0.0)
            n = i + 1
            sum_metrics["loss"] += batch_loss
            results = {metric: value / n for metric, value in sum_metrics.items()}
            pbar.update()
            pbar.set_postfix_str(" ".join(f"{met}={val:.3f}" for met, val in results))
        return results

    def _val_step(
        self,
        model: TransformerModel,
        val: DataLoader,
        loss_fn: torch.nn.Module,
        pbar: tqdm,
        expanded_metrics: bool = False,
    ) -> Dict[str, float]:
        all_outputs = []
        all_targets = []
        with torch.no_grad():
            model.eval()
            for x, mask, targets in val:
                outputs = model(input_ids=x, attention_mask=mask)
                if self.task == Task.MULTICLASS_SEQUENCE:
                    outputs, targets = apply_mask(
                        outputs, targets, mask, n_labels=outputs.shape[-1]
                    )
                all_outputs.extend(outputs)
                all_targets.extend(targets.tolist())
                pbar.update()
        if not all_outputs:
            return []
        processor = self.task.output_processor()
        if self.activation_fn:
            processor = partial(processor, activation=self.activation_fn)
        loss, metrics = processor(
            torch.stack(all_outputs), 
            torch.tensor(all_targets), 
            loss_fn,
            expanded_metrics=expanded_metrics,
        )
        return {"loss": loss, **metrics}
