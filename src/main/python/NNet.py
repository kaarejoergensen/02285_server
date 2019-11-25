import time
import torch
from torch.utils.data import DataLoader

from NNetModule import NNetModule
import numpy as np

from StateDataSet import StateDataSet


class NNet():
    def __init__(self):
        self.filepath = "model.pth" #TODO: Ikke hardkode
        self.model =NNetModule()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = torch.nn.BCELoss()


    #State skal ta inn staten + scoren gitt fra mcts

    # Init NN
    # Kjøre en gang MCTS, lagre alle states med socres
    # Train nettverk på denne dataoen
    # Kjøre MCTS en gang til for valideringsett
    # Teste accuracien til nettverket
    def train(self, states, scores, epoch=2, batch_size=2):
        #Omgjør states og scores til numpy arrays
        trainSet = StateDataSet(states, scores)
        train_loader = DataLoader(dataset=trainSet, batch_size=batch_size, shuffle=True, num_workers=0)

        running_loss = 0.0
        for batch in train_loader:
            input, labels = batch
            self.optimizer.zero_grad()
            score_predication = self.model(input)

            loss = self.criterion(score_predication, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

        print("Training session complete...")
        print(loss)


    def predict(self, state):
        return self.model(torch.tensor(np.array(state), dtype=torch.float))

    def save_checkpoint(self):
        torch.save(self.model.state_dict(), self.filepath)


    def load_checkpoint(self, filepath):
        self.nnetModule = self.model.load_state_dict(torch.load(self.filepath))


    def close(self):
        pass
