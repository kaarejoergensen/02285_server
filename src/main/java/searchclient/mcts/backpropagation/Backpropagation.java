package searchclient.mcts.backpropagation;

import searchclient.mcts.model.Node;

public interface Backpropagation {
    void backpropagate(float score, Node nodeToExplore, Node root);
}
