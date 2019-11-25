import torch
import numpy as np
from torch.utils.data import Dataset


class StateDataSet(Dataset):
    def __init__(self, states, scores):
        npStates = np.array(states)
        self.len = npStates.size
        self.states = torch.tensor(npStates, dtype=torch.float)
        self.scores = torch.tensor(np.array(scores), dtype=torch.float)


    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.states[item], self.scores[item]




