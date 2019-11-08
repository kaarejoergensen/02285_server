package searchclient.mcts.backpropagation;

import searchclient.mcts.Node;

public interface Backpropagation {
    void backpropagate(int score, Node nodeToExplore, Node root);
}
