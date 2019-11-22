import time

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
