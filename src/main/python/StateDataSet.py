import numpy as np
import torch
from torch.utils.data import Dataset


class StateDataSet(Dataset):
    def __init__(self, states, probability_vectors, wins):
        self.len = len(states)
        self.states = torch.tensor(np.array(states).astype(np.float64), dtype=torch.float)
        self.probability_vectors = torch.tensor(np.array(probability_vectors).astype(np.float64), dtype=torch.float)
        self.win = torch.tensor(np.array(wins).astype(np.float64), dtype=torch.float)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.states[item], self.probability_vectors[item], self.win[item]




