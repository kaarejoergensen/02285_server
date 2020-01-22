import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from NNetAlphaModule import NNetAlphaModule, AlphaLoss


class NNet():
    def __init__(self, gpu):
        #self.model = NNetModule()
        self.device = torch.device('cuda:' + str(gpu))
        self.model = NNetAlphaModule(resblocks=19)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            self.model = self.model.to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=10e-4)
        self.criterion = AlphaLoss()

    # State skal ta inn staten + scoren gitt fra mcts

    # Init NN
    # Kjøre en gang MCTS, lagre alle states med socres
    # Train nettverk på denne dataoen
    # Kjøre MCTS en gang til for valideringsett
    # Teste accuracien til nettverket
    def train(self, trainSet, epochs=20, batch_size=64):
        running_loss = 0.0
        for epoch in range(epochs):
            self.model.train()
            train_loader = DataLoader(dataset=trainSet, batch_size=batch_size, shuffle=True, drop_last=True)

            for batch in train_loader:
                states, probability_vectors, wins = batch
                if torch.cuda.is_available():
                    states, probability_vectors, wins = \
                        states.to("cuda"), probability_vectors.to("cuda"), wins.to("cuda")
                    states, probability_vectors, wins = \
                        states.to(self.device), probability_vectors.to(self.device), wins.to(self.device)
                self.optimizer.zero_grad()
                out_probability_vectors, out_wins = self.model(states)
                total_loss = self.criterion(out_wins[:, 0], wins, out_probability_vectors, probability_vectors)
                total_loss.backward()
                self.optimizer.step()
                running_loss += total_loss.item()

        return running_loss / epochs

    def predict(self, state):
        state_tensor = torch.tensor(np.array(state), dtype=torch.float)
        if torch.cuda.is_available():
            state_tensor = state_tensor.to("cuda")
            state_tensor = state_tensor.to(self.device)
        self.model.eval()
        with torch.no_grad():
            probability_vector, win = self.model(state_tensor)

        return torch.exp(probability_vector).data.cpu().numpy()[0].tolist(), win.data.cpu().numpy()[0].tolist()

    def save_checkpoint(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load_checkpoint(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage))

    def close(self):
        pass
