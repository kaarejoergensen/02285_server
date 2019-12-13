import ast
import sys
import threading
import time
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


def main():
    server_messages = sys.stdin
    parser = Main(server_messages)
    nnet = NNet()
    while True:
        lines = parser.receive()
        method = lines.pop(0)
        if method == "train":
            states = []
            probability_vectors = []
            wins = []
            for line in lines:
                state, probability_vector, won = line.split("|")
                states.append(ast.literal_eval(state))
                probability_vectors.append(ast.literal_eval(probability_vector))
                wins.append(won)
            train_set = StateDataSet(states, probability_vectors, wins)
            flush_print(nnet.train(train_set))
        elif method == "predict":
            state = ast.literal_eval(lines[0])
            probability_vector, win = nnet.predict(state)
            flush_print(probability_vector)
            flush_print(win)
            flush_print(0)
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
    main()
