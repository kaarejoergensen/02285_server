import ast
import sys
import threading
import time
from argparse import ArgumentParser
from queue import Queue

from NNet import NNet
import numpy as np
from StateDataSet import StateDataSet


def flush_print(msg):
    print(msg, flush=True)


def stderr_print(msg):
    print(msg, file=sys.stderr, flush=True)


class Main:
    def __init__(self, server_messages):
        self.server_messages = server_messages

    def receive(self):
        lines = []
        line = self.server_messages.readline().rstrip()
        while line and line != "done":
            # stderr_print(line)
            lines.append(line)
            line = self.server_messages.readline().rstrip()
        if len(lines) == 0:
            stderr_print("No lines received")
            exit(-1)
        return lines


def main(args):
    server_messages = sys.stdin
    parser = Main(server_messages)
    nnet = NNet(args)
    while True:
        lines = parser.receive()
        method = lines.pop(0)
        if method == "train":
            states = []
            probability_vectors = []
            wins = []
            for line in lines:
                state, probability_vector, won = line.split("|")
                try:
                    states.append(ast.literal_eval(state))
                    probability_vectors.append(ast.literal_eval(probability_vector))
                except ValueError:
                    stderr_print("Error parsing line")
                    stderr_print(line)
                    raise ValueError
                wins.append(won)
            train_set = StateDataSet(states, probability_vectors, wins)
            flush_print(nnet.train(train_set, epochs=args.epochs, batch_size=args.batch_size, print_loss=args.print_loss))
        elif method == "predict":
            try:
                state = ast.literal_eval(lines[0])
            except ValueError:
                stderr_print("Error parsing line")
                stderr_print(lines[0])
                raise ValueError
            probability_vector, win = nnet.predict(state)
            flush_print(probability_vector)
            flush_print(win)
        elif method == "saveModel":
            nnet.save_checkpoint(lines[0])
            flush_print("done")
        elif method == "loadModel":
            nnet.load_checkpoint(lines[0])
            flush_print("done")
        elif method == "close":
            nnet.close()
            flush_print("done")
            break
        else:
            stderr_print("Unknown method: " + method)
            exit(-1)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--gpu", type=float, default=0, help="GPU to use")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--resblocks", type=int, default=19, help="Number of resblocks")
    parser.add_argument("--print_loss", type=bool, default=False, help="Print loss at each epoch")
    parser.add_argument("--loss_function", default="MSE",
                        choices=["MSE", "MAE"], help="Choose loss function")

    args = parser.parse_args()
    main(args)
