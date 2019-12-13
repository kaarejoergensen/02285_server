import sys

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from NNetModule import NNetModule


class NNet():
    def __init__(self):
        self.model = NNetModule()
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            # if torch.cuda.device_count() > 1:
                # print("Using", torch.cuda.device_count(), "GPUs", file=sys.stderr, flush=True)
                # print("", file=sys.stderr, flush=True)
                # self.model = nn.DataParallel(self.model)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.criterion = torch.nn.BCELoss()
        self.priorLoss = 1.0

    # State skal ta inn staten + scoren gitt fra mcts

    # Init NN
    # Kjøre en gang MCTS, lagre alle states med socres
    # Train nettverk på denne dataoen
    # Kjøre MCTS en gang til for valideringsett
    # Teste accuracien til nettverket
    def train(self, trainSet, epochs=20, batch_size=64):
        # copy = kopi.deepcopy(self.model)
        # Omgjør states og scores til numpy arrays
        for epoch in range(epochs):
            self.model.train()
            train_loader = DataLoader(dataset=trainSet, batch_size=batch_size, shuffle=True, drop_last=True)

            running_loss = 0.0
            for batch in train_loader:
                states, probability_vectors, wins = batch
                if torch.cuda.is_available():
                    states = states.to("cuda")
                    probability_vectors = probability_vectors.to("cuda")
                    wins = wins.to("cuda")
                self.optimizer.zero_grad()
                self.model.eval()
                out_probability_vectors, out_wins = self.model(states)
                # labels = labels.view(-1, 1)
                loss_pv = self.loss_pv(probability_vectors, out_probability_vectors)
                loss_w = self.loss_w(wins, out_wins)
                total_loss = loss_pv + loss_w

                #loss = self.criterion(score_predication, labels)
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                running_loss += total_loss.item()

            # if(loss.item() > self.priorLoss):
            #   self.model = copy
            #  return self.priorLoss

            # self.priorLoss = loss.item()
        return running_loss / epochs

    def loss_pv(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_w(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def predict(self, state):
        state_tensor = torch.tensor(np.array(state), dtype=torch.float)
        if torch.cuda.is_available():
            state_tensor = state_tensor.to("cuda")
        self.model.eval()
        with torch.no_grad():
            probability_vector, win = self.model(state_tensor)

        return torch.exp(probability_vector).data.cpu().numpy()[0].tolist(), win.data.cpu().numpy()[0].tolist()

    def save_checkpoint(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load_checkpoint(self, filepath):
        self.model.load_state_dict(torch.load(filepath))

    def close(self):
        pass
