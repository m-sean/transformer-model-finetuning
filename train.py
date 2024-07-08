import argparse
import os
from configparser import ConfigParser

from tmt.optimizer import OptimizerConfig
from tmt.handler import TransformerModelHandler
from tmt.task import Task

OPTIMIZERS = {
    "adam": OptimizerConfig.Adam,
    "adamw": OptimizerConfig.AdamW,
    "sgd": OptimizerConfig.SGD,
}


def main(args: argparse.Namespace) -> None:
    config = ConfigParser()
    config.read(args.config)
    task = Task.from_str(config["model"]["task"])

    label_mapping = None
    output_activation_fn = None
    if task != Task.REGRESSION:
        labels = [label.strip() for label in config["data"]["labels"].split(",")]
        label_mapping = {lbl: idx for idx, lbl in enumerate(labels)}
    else:
        output_activation_fn = config.get("model", "output_activation", fallback="none")

    model_trainer = TransformerModelHandler(
        model_name=config["model"]["name"],
        task=task,
        label_mapping=label_mapping,
        batch_size=config.getint("training", "batch_size"),
        activation_fn=output_activation_fn,
    )
    x_field = config["data"]["x_field"]
    y_field = config["data"]["y_field"]

    train_loader = model_trainer.get_data_loader(
        config["data"]["train"], train=True, x_field=x_field, y_field=y_field
    )
    if dev := config.get("data", "dev", fallback=None):
        val_loader = model_trainer.get_data_loader(
            dev, train=False, x_field=x_field, y_field=y_field
        )
    else:
        val_loader = model_trainer.empty_loader()

    test_loader = None
    if test := config.get("data", "test", fallback=None):
        test_loader = model_trainer.get_data_loader(
            test, train=False, x_field=x_field, y_field=y_field
        )

    optimizer = OPTIMIZERS[config.get("training", "optimizer")]
    optimizer.set_lr(config.getfloat("training", "learning_rate"))

    if beta_1 := config["training"].getfloat("beta_1"):
        optimizer.set_betas((beta_1, config.getfloat("training", "beta_2")))
    if eps := config.getfloat("training", "eps"):
        optimizer.set_eps(eps)
    if decay := config.getfloat("training", "decay"):
        optimizer.set_decay(decay)
    if amsgrad := config.getboolean("training", "amsgrad"):
        optimizer.set_amsgrad(amsgrad)

    save_dir = config["model"]["save_dir"]
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    epochs = config.getint("training", "epochs")
    model_trainer.train(
        train_loader,
        val_loader,
        test=test_loader,
        epochs=epochs,
        save=save_dir,
        optimizer=optimizer,
        norm_clip=config.getfloat("training", "norm_clip", fallback=None),
        loss=config["training"]["loss"],
        save_top=config.getint("training", "save_top_n_models", fallback=epochs),
        scaler=config.getboolean("training", "half_precision"),
        dropout=config.getfloat("training", "dropout", fallback=0.0)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", type=str, help="`.ini` config file with training hyperparameters"
    )
    main(parser.parse_args())
