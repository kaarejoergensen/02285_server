package searchclient.mcts.backpropagation.impl;

import searchclient.mcts.Node;
import searchclient.mcts.backpropagation.Backpropagation;

public class AdditiveBackpropagation implements Backpropagation {
    private static final int WIN_SCORE = 10;

    @Override
    public void backpropagate(Node nodeToExplore, boolean solved) {
        Node tempNode = nodeToExplore;
        while (tempNode != null) {
            tempNode.incrementVisitCount();
            if (solved) tempNode.addScore(WIN_SCORE);
            tempNode = tempNode.getParent();
        }
    }
}
