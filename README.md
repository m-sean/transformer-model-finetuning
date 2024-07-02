## Transformer Model Training

A repo for fine-tuning transformer models like RoBERTa for various modeling tasks.

### Training

To train a model, set up a new `.ini` config file with the required inputs. Then run
```
python3 train.py <<PATH TO CONFIG>>
```

#### [DATA]
Datasets must be in `jsonl` or `csv` format. 
CSVs should have column headers to identify the input texts and the labels/targets.
<p>For multilabel tasks, the label field should be a list/array or list-like string value (stringified python list, or comma-separated string value).</p>
<p>For sequence labeling tasks (e.g. NER), the files should be <code>.jsonl</code> with one JSON formatted record per line. Each record should have arrays contained within the x and y fields. Note that transformer tokenizers may split tokens, so extra preproccessing of the labels may be necessary. (TODO: Make a script for this)</p>

#### [MODEL]
Supported tasks: `binary`, `multiclass`, `multiclass_sequence`, `multilabel` or `regression`.
<p>The training script will log metrics for each epoch and save a model checkpoint with the epoch number which can later be used to specify the model for export.</p>
Regression models can additionally have an `output_activation` function specified in the config: `sigmoid`, `tanh`, `relu`, and `identity` are all supported. More are available in the [torch docs](https://pytorch.org/docs/stable/nn.html#module-torch.nn). See `tmt/util.py` to expand/customize.

#### [TRAINING]
Specify number of `epochs`, `batch_size`, `loss` function and other training hyperparameters here.
Each task is optimized over a correlation metric (mcc, r_k, or r^2) and `save_top_n_models` specifies how many model checkpoints you want to save based on this metric (default is 1).
Use of `half_precision` is only recommended when training on a GPU enabled machine. 
<br>**loss functions:** `bce`, `cce`, `lsce` (label-smoothing), `focal`, `huber`, and `mse`.

### Exporting
To export a model, specify the checkpoint number (e.g. "e4") in the `use_checkpoint` field under `[model]` in your confing.ini and run:
```
python3 export-model.py torch <<PATH TO CONFIG>>
```
or, for an onnx model:
```
python3 export-model.py onnx <<PATH TO CONFIG>>
```
**NOTE:** Not all transformer architectures are compatible with both/either of these export formats.
