import torch.optim as optim

from torch import nn
import torch
import filehandler
from nettverk import Nettverk
from torch.utils.data import DataLoader
import numpy as np
from matchData import MatchDataset


class trainer:
    def __init__(self, net):
        # MSE loss, mean square error
        # BCE loss, binary cross entropy loss, gir svar mellom 0 og 1
        self.criterion = torch.nn.BCELoss()
        #self.criterion = nn.CrossEntropyLoss()

        # lav loss betyr mest tilnærmet datasettet -> mer accuracy
        # sgd er en annen lossfunksjon som har moment
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        #self.optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        #self.path = './'
        self.net = net



    def accuracy(self, y_pred, labels):
        correct = 0
        for i, value in enumerate(y_pred):
            if((value > 0.5) & (labels[i] == 1)):
                correct += 1
            if ((value < 0.5) & (labels[i] == 0)):
                correct += 1
        return (correct / len(labels))*100



    #Epoch definerer hvor mange ganger vi skal kjøre datasettet
    def trainNetwork(self, epoch=3, train_loader=None, validation_loader=None):
        for e in range(epoch):
            running_loss = 0.0
            i = 0
            for batchdata in train_loader:
                # get the inputs; data is a list of [inputs, labels]
                labels, inputs = batchdata
                # zero the parameter gradients
                self.optimizer.zero_grad()

                y_pred = model(inputs)

                # forward + backward + optimize
                #outputs = self.net(temp_heroes)
                loss = se lf.criterion(y_pred, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if(i % 1000 == 0):
                    self.validaiton(validation_loader)
                    #print(i, "/ 10000")
                    print(loss)
                    print(self.accuracy(y_pred=y_pred, labels=labels))
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
                i += 1
                #print('Finished Training')
        i = 2
        # torch.save(self.net.state_dict(), self.path)

    def validaiton(self, validation_loader=None):
        for batchdata in validation_loader:
            # get the inputs; data is a list of [inputs, labels]
            labels, inputs = batchdata
            # zero the parameter gradients

            y_pred = model(inputs)

            # forward + backward + optimize
            # outputs = self.net(temp_heroes)
            loss = self.criterion(y_pred, labels)
            print("VALIDATION: ")
            print(self.accuracy(y_pred, labels))
            print("###########################")






#Starte med å importere dataen
allDataWins, allDataHeroes = filehandler.load_into_matrix()
allDataWins = np.array(allDataWins)
allDataHeroes = np.array(allDataHeroes)

trainSet = MatchDataset(allDataWins[0:10000],allDataHeroes[0:10000])
train_loader = DataLoader(dataset=trainSet, batch_size=32, shuffle=True, num_workers=0)


validationSet = MatchDataset(allDataWins[10000:11000],allDataHeroes[10000:11000])
validation_loader = DataLoader(dataset=validationSet, batch_size=1000, shuffle=True, num_workers=0)


model = Nettverk()
try:
    model.to('cuda')
    print("Cuda enabeled. Let's go!")
except:
    print("CUDA not compatible with GPU - training regularly")

t = trainer(model)

t.trainNetwork(epoch=1000, train_loader=train_loader, validation_loader=validation_loader)
print("--- TRAINING COMPLETE ---")

