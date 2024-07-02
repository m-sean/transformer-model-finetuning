from typing import Dict
import json
import os
import torch
import torch.nn as nn
from transformers import AutoModel


class TransformerModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        input_dim: int,
        output_dim: int,
        task: str,
        dropout: float = 0.0,
    ) -> None:
        super(TransformerModel, self).__init__()
        self.base = AutoModel.from_pretrained(
            model_name, 
            return_dict=False, 
            output_hidden_states=True
        )
        self.dropout = nn.Dropout(p=dropout)
        self.output = nn.Linear(self.base.config.hidden_size, output_dim)
        self.config = {
            "model_name": model_name,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "task": task,
        }

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        x = self.base(input_ids, attention_mask=attention_mask)[0]
        x = self.dropout(x)
        x = x * torch.unsqueeze(attention_mask, 2)
        x = torch.sum(x, 1) / torch.unsqueeze(torch.sum(attention_mask, 1), 1)
        x = self.output(x)
        return x

    def trainable_params(self):
        return [param for _, param in self.named_parameters()]

    def _save_config(self, dir):
        with open(f"{dir}/config.json", "w") as fp:
            json.dump(self.config, fp)

    @staticmethod
    def _load_config(dir) -> Dict[str, any]:
        with open(f"{dir}/config.json", "r") as fp:
            return json.load(fp)

    def save(self, dir: str, checkpoint: str = "") -> None:
        if not os.path.isdir(dir):
            os.mkdir(dir)
        self._save_config(dir)
        if checkpoint:
            checkpoint = f"_{checkpoint}"
        torch.save(self.state_dict(), f=f"{dir}/model{checkpoint}.pt")

    @classmethod
    def load(cls, dir: str, checkpoint: str = ""):
        config = cls._load_config(dir)
        model = cls(**config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if checkpoint:
            state_dict = torch.load(f"{dir}/model_{checkpoint}.pt", map_location=device)
        else:
            state_dict = torch.load(f"{dir}/model.pt", map_location=device)
        model.load_state_dict(state_dict)
        return model


class TransformerSequenceModel(TransformerModel):

    def __init__(self, *args, **kwargs) -> None:
        super(TransformerSequenceModel, self).__init__(*args, **kwargs)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        x = self.base(input_ids, attention_mask=attention_mask)[0]
        x = self.dropout(x)
        x = self.output(x)
        return x
