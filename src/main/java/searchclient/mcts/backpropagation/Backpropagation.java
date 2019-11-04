package searchclient.mcts.backpropagation;

import searchclient.mcts.Node;

public interface Backpropagation {
    void backpropagate(Node nodeToExplore, boolean solved);
}
