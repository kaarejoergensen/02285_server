import ast
import sys

from NNet import NNet


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
        while line:
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
            examples = []
            for line in lines:
                examples.append(ast.literal_eval(line))
            nnet.train(examples)
            flush_print("done")
        elif method == "predict":
            state = []
            for line in lines:
                state.append(ast.literal_eval(line))
            flush_print(nnet.predict(state))
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
