import argparse
import configparser
import onnx, onnxoptimizer
import os
import torch

from convolve.model.components import NeuralComponents
from convolve.model import (
    LabelIndexer, SequenceScoreForm, Task as ConvolveTask,
)

from tmt.model import TransformerModel, TransformerSequenceModel
from tmt.handler import TransformerModelHandler
from tmt.task import Task


def export_onnx(
    export_dir: str,
    model: TransformerModel,
    inputs: tuple[torch.Tensor, torch.Tensor],
):
    output_axes = {0: "batch_size"}
    if model.config["task"] == "multiclass_sequence":
        output_axes[1] = "length"

    onnx_path = f"{export_dir}/model.onnx"
    torch.onnx.export(
        model=model,
        args=inputs,
        f=onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["output"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "length"},
            "attention_mask": {0: "batch_size", 1: "length"},
            "output": output_axes,
        },
    )
    onnx_model = onnx.load(onnx_path)
    model = onnxoptimizer.optimize(onnx_model)
    with open(f"{export_dir}/model.opt.onnx", "wb") as file:
        file.write(model.SerializeToString())


def export_torch(
    export_dir: str,
    model: TransformerModel,
    inputs: tuple[torch.Tensor, torch.Tensor],
):
    scripted_model = torch.jit.trace(model, example_inputs=inputs)
    torch.jit.save(scripted_model, f"{export_dir}/model.pt")


_MODEL_EXPORTERS = {
    "torch": export_torch,
    "onnx": export_onnx,
}


def main(args: argparse.Namespace) -> None:

    config = configparser.ConfigParser()
    config.read(args.config)

    task = Task.from_str(config["model"]["task"])
    model_name = config["model"]["name"]
    export_dir = config["model"]["export_dir"]

    if not os.path.isdir(export_dir):
        os.mkdir(export_dir)

    label_mapping = None
    if task != Task.REGRESSION:
        labels = [label.strip() for label in config["data"]["labels"].split(",")]
        label_mapping = {lbl: idx for idx, lbl in enumerate(labels)}

    handler = TransformerModelHandler(
        model_name=model_name,
        task=task,
        label_mapping=label_mapping,
        batch_size=config.getint("training", "batch_size"),
    )

    loader = handler.get_data_loader(
        config["data"]["train"],
        train=False,
        x_field=config["data"]["x_field"],
        y_field=config["data"]["y_field"],
    )

    model_cls = (
        TransformerModel
        if task != Task.MULTICLASS_SEQUENCE
        else TransformerSequenceModel
    )
    model = model_cls.load(
        config["model"]["save_dir"], checkpoint=config["model"]["use_checkpoint"]
    )

    model.eval()
    model.to(handler.device)

    if task == Task.REGRESSION:
        task = ConvolveTask.regression
    else:
        label_idxr = LabelIndexer.from_ordered_labels(labels)
        if task == Task.BINARY:
            task = ConvolveTask.binary_class(label_idxr)
        elif task == Task.MULTICLASS:
            task = ConvolveTask.multi_class(label_idxr)
        elif task == Task.MULTICLASS_SEQUENCE:
            task = ConvolveTask.multi_class_sequence(
                label_idxr,
                SequenceScoreForm.vector,
                max_seq_len=handler.max_input,
            )

    try:
        handler.tokenizer.save_pretrained(f"{export_dir}/tokenizer")
        components = NeuralComponents.transformer(
            "",
            task=task,
            add_special_tokens=False,
            pad_token_idx=handler.tokenizer.pad_token_id,
        )
        components.save(f"{export_dir}/components.json")
            
    except Exception as err:
        print(f"Unable to write `components.json`, skipping: {err}")


    input_ids, attention_mask, _ = next(iter(loader))
    exporter = _MODEL_EXPORTERS[args.export_type]
    exporter(export_dir, model, inputs=(input_ids, attention_mask))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("export_type", choices=["onnx", "torch"])
    parser.add_argument("config", type=str, help="`.ini` config file used for training")
    main(parser.parse_args())
