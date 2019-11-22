import sys
import time

import torch

from NNetModule import NNetModule


class NNet:
    def __init__(self):
        self.nnetModule = NNetModule()

    def train(self, examples):
        pass

    def predict(self, state):
        time.sleep(2)
        return 0.0

    def save_checkpoint(self, filepath):
        pass

    def load_checkpoint(self, filepath):
        pass

    def close(self):
        pass

    def info(self):
        print(torch.cuda.current_device(), file=sys.stderr, flush=True)
        print(torch.cuda.device_count(), file=sys.stderr, flush=True)
        print(torch.cuda.get_device_name(), file=sys.stderr, flush=True)
        print(torch.cuda.is_available(), file=sys.stderr, flush=True)