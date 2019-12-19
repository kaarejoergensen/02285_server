package searchclient.mcts.backpropagation;

import searchclient.mcts.model.Node;

public abstract class Backpropagation implements Cloneable {
    public abstract void backpropagate(float score, Node nodeToExplore, Node root);

    @Override
    public abstract Backpropagation clone();
}
