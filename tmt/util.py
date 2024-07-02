import convem
import torch
from torcheval.metrics.functional import r2_score
from typing import List, Optional
from tmt.loss import FocalLoss, LabelSmoothingCrossEntropy

_REGRESSION_ACTIVATION_FNS = {
    "sigmoid": torch.nn.Sigmoid(),
    "tanh": torch.nn.Tanh(),
    "relu": torch.nn.ReLU(),
    "identity": torch.nn.Identity(),
}


def apply_mask(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    n_labels: Optional[int] = None,
):
    active = mask.view(-1) == 1
    if n_labels:
        logits = logits.view(-1, n_labels)[active]
    else:
        logits = logits.view(-1)[active]
    targets = targets.view(-1)[active]
    return logits, targets


def binary_task_output(
    logits: torch.Tensor, 
    targets: torch.Tensor, 
    loss_fn: torch.nn.Module, 
    expanded_metrics: bool = False,
):
    preds = logits.sigmoid().squeeze()
    loss = loss_fn(preds, targets)
    targets = targets == 1.0
    confusion_matrix = convem.binary_confusion_matrix(preds.tolist(), targets.tolist())
    metrics = {"acc": confusion_matrix.accuracy(), "mcc": confusion_matrix.mcc()}
    if expanded_metrics:
        print(confusion_matrix)
        metrics["pre"] = confusion_matrix.precision()
        metrics["rec"] = confusion_matrix.recall()
        metrics["f1s"] = confusion_matrix.f1()
        metrics["ksc"] = confusion_matrix.k_score()
    return loss, metrics


def multiclass_task_output(
    logits: torch.Tensor, 
    targets: torch.Tensor, 
    loss_fn: torch.nn.Module,
    expanded_metrics: bool = False,
):
    preds = logits.softmax(-1)
    if isinstance(loss_fn, LabelSmoothingCrossEntropy) or isinstance(loss_fn, FocalLoss):
        loss = loss_fn(logits, targets)
    else:
        loss = loss_fn(preds, targets)
    confusion_matrix = convem.multi_confusion_matrix(preds.tolist(), targets.tolist())
    metrics = {"acc": confusion_matrix.accuracy(), "rk": confusion_matrix.rk()}
    if expanded_metrics:
        print(confusion_matrix)
        metrics["w_pre"] = confusion_matrix.precision("weighted")
        metrics["w_rec"] = confusion_matrix.recall("weighted")
        metrics["w_f1s"] = confusion_matrix.f1("weighted")
        metrics["ksc"] = confusion_matrix.k_score()
    return loss, metrics

def multilabel_task_output(
    logits: torch.Tensor, 
    targets: torch.Tensor, 
    loss_fn: torch.nn.Module,
    expanded_metrics: bool = False,
):
    metrics = {}
    sum_inv_pos_mcc = 0
    finite_count = 0
    _, n_labels = logits.shape
    preds = logits.sigmoid()
    loss = loss_fn(preds, targets)
    targets = targets == 1.0
    for idx in range(n_labels):
        bcm = convem.binary_confusion_matrix(
            preds[:, idx].tolist(), targets[:, idx].tolist()
        )
        mcc = bcm.mcc()
        if mcc is not None:
            sum_inv_pos_mcc += 1 / (mcc or 1e-3)
            finite_count += 1
        if expanded_metrics:
            metrics[idx] = {
                "mcc": mcc,
                "pre": bcm.precision(),
                "rec": bcm.recall(),
                "f1s": bcm.f1(),
            }
    harmonic_mean = finite_count / sum_inv_pos_mcc if sum_inv_pos_mcc > 0 else 0
    metrics["Hmcc"] = harmonic_mean
    return loss, metrics


def regression_task_output(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: torch.nn.Module,
    activation: str = "identity",
    expanded_metrics: bool = False,
):
    preds = _REGRESSION_ACTIVATION_FNS[activation](logits).squeeze()
    loss = loss_fn(preds, targets)
    metrics = {"r2": r2_score(preds, targets)}
    return loss, metrics


def pad_sequence(sequence: List[any], max_len: int, pad_val: any):
    if isinstance(sequence, tuple):
        sequence = list(sequence)
    if len(sequence) > max_len:
        return sequence[:max_len]
    for _ in range(max_len - len(sequence)):
        sequence.append(pad_val)
    return sequence
