import numpy as np
import torch
from torch.utils.data import Dataset


class StateDataSet(Dataset):
    def __init__(self, states, scores):
        self.len = len(states)
        npStates = np.array(states)
        self.states = torch.tensor(npStates, dtype=torch.float)
        self.scores = torch.tensor(np.array(scores).flatten(), dtype=torch.float)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.states[item], self.scores[item]




