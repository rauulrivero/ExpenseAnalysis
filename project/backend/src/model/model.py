import torch
import torch.nn as nn
import numpy as np

class DigitalTwinModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        return self.net(x)
    

class SingleOutputModel(nn.Module):
    """
    Small MLP that predicts one output, with positive enforcement via Softplus.
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )

    def forward(self, x):
        # x: [batch, in_dim]
        return self.net(x)  # output: [batch, 1]

class EnsembleDigitalTwin(nn.Module):
    """
    Wraps 45 independent SingleOutputModel instances,
    one per output category. Forward pass runs all submodels.
    """
    def __init__(self, in_dim: int, n_outputs: int = 45):
        super().__init__()
        # Create a list of submodels
        self.submodels = nn.ModuleList(
            [SingleOutputModel(in_dim) for _ in range(n_outputs)]
        )

    def forward(self, x):
        # x: [batch, in_dim]
        # Collect outputs from each submodel
        outputs = []
        for sub in self.submodels:
            # each out_i is shape [batch, 1]
            outputs.append(sub(x))
        # Concatenate to [batch, n_outputs]
        return torch.cat(outputs, dim=1)
