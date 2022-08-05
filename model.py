from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 

def get_hidden_limits(layer: nn.Linear) -> Tuple[float, float]:
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """
    Actor (policy) model.
    """
    def __init__(
        self, 
        state_size: int, 
        action_size: int, 
        seed: int, 
        fc1_units: int = 128,
        fc2_units: int = 128,
    ) -> None:
        """
        Initialize model.

        Params
        ======
        state_size: dimension of state
        action_size: dimension of action
        seed: random seed
        fc1_units: number of nodes in the first hidden layer
        fc2_units: number of  nodes in the second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Reset model parameters.
        """
        self.fc1.weight.data.uniform_(*get_hidden_limits(self.fc1))
        self.fc2.weight.data.uniform_(*get_hidden_limits(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: np.ndarray) -> torch.Tensor:
        """
        Map state to action.

        Params
        ======
        state

        Returns
        =======
        action
        """
        if state.dim() == 1:
            state.unsqueeze_(0)
        x = F.relu(self.fc1(state))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """
    Critic (value) model.
    """
    def __init__(
        self, 
        state_size: int, 
        action_size: int, 
        seed: int,
        fcs1_units: int = 128, 
        fc2_units: int = 128,
    ) -> None:
        """
        Initialize critic model.

        Params
        ======
        state_size: dimension of state
        action_size: dimension of action
        seed: random seed
        fcs1_units: number of nodes in the first hidden layer
        fc2_units: number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)

        self.bn1 = nn.BatchNorm1d(fcs1_units)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Reset model parameters.
        """
        self.fcs1.weight.data.uniform_(*get_hidden_limits(self.fcs1))
        self.fc2.weight.data.uniform_(*get_hidden_limits(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: np.ndarray, action: torch.Tensor) -> torch.Tensor:
        """
        Map state and action to Q values.

        Params
        ======
        state
        action

        Returns
        =======
        Q-value
        """
        if state.dim() == 1:
            state.unsqueeze_(0)
        xs = F.relu(self.fcs1(state))
        xs = self.bn1(xs)
        x = torch.cat((xs, action.float()), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)