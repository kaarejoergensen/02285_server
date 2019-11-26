import numpy as np
import torch
from torch.utils.data import DataLoader

from NNetModule import NNetModule


class NNet():
    def __init__(self):
        self.filepath = "model.pth"  # TODO: Ikke hardkode
        self.model = NNetModule()
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = torch.nn.BCELoss()
        self.priorLoss = 1.0

    # State skal ta inn staten + scoren gitt fra mcts

    # Init NN
    # Kjøre en gang MCTS, lagre alle states med socres
    # Train nettverk på denne dataoen
    # Kjøre MCTS en gang til for valideringsett
    # Teste accuracien til nettverket
    def train(self, trainSet, epoch=2, batch_size=256):
        # copy = kopi.deepcopy(self.model)
        # Omgjør states og scores til numpy arrays
        #trainSet = StateDataSet(states, scores)
        train_loader = DataLoader(dataset=trainSet, batch_size=batch_size, shuffle=True)

        running_loss = 0.0
        for batch in train_loader:
            input, labels = batch
            if torch.cuda.is_available():
                input = input.to("cuda")
                labels = labels.to("cuda")
            self.optimizer.zero_grad()
            score_predication = self.model(input)
            labels = labels.view(-1, 1)
            loss = self.criterion(score_predication, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

        # if(loss.item() > self.priorLoss):
        #   self.model = copy
        #  return self.priorLoss

        # self.priorLoss = loss.item()
        return loss.item()

    def predict(self, state):
        stateTensor = torch.tensor(np.array(state), dtype=torch.float)
        if torch.cuda.is_available():
            stateTensor = stateTensor.to("cuda")
        return self.model(stateTensor)

    def save_checkpoint(self):
        torch.save(self.model.state_dict(), self.filepath)

    def load_checkpoint(self, filepath):
        self.nnetModule = self.model.load_state_dict(torch.load(self.filepath))

    def close(self):
        pass
