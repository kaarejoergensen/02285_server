import ast
import sys
import threading
from queue import Queue

from NNet import NNet
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
            lines.append(line)
            line = self.server_messages.readline().rstrip()
        if len(lines) == 0:
            stderr_print("No lines received")
            exit(-1)
        return lines


def main():
    server_messages = sys.stdin
    parser = Main(server_messages)
    q = Queue()
    nnet = NNet()
    workerThread = threading.Thread(target=worker, args=(q, nnet))
    workerThread.start()
    while True:
        lines = parser.receive()
        method = lines.pop(0)
        if method == "train":
            # stderr_print("Received train call")
            states = ast.literal_eval(lines[0])
            scores = ast.literal_eval(lines[1])
            trainSet = StateDataSet(states, scores)
            q.put(trainSet)
            # flush_print(nnet.train(states, scores))
            # stderr_print("Parsing done")
        elif method == "predict":
            state = ast.literal_eval(lines[0])
            output = nnet.predict(state).item()
            flush_print(output)
        elif method == "saveModel":
            nnet.save_checkpoint(lines[0])
            flush_print("done")
        elif method == "loadModel":
            nnet.load_checkpoint(lines[0])
            flush_print("done")
        elif method == "close":
            nnet.close()
            q.put(None)
            workerThread.join()
            flush_print("done")
            break
        else:
            stderr_print("Unknown method: " + method)
            exit(-1)


def worker(queue, nnet):
    while True:
        item = queue.get(block=True, timeout=None)
        if item is None:
            break
        # stderr_print("Training")
        flush_print(nnet.train(item))
        # stderr_print("Trained!")


if __name__ == '__main__':
    main()
