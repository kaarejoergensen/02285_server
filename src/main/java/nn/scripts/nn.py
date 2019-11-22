from torch import nn

class NN(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        #Som test så bruker vi 5x5 maps, med 3 layers ergo 75
        self.hidden1 = nn.Linear(75, 256)
        self.hidden2 = nn.Linear(256, 10)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(10, 1)


        # Define sigmoid activation and softmax output
        self.sigmoid = nn.Sigmoid()
        ##Softmax runner svaret til å være mellom 0 og 1
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden1(x)
        x = self.sigmoid(x)
        x = self.hidden2(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x