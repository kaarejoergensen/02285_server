package searchclient.mcts.backpropagation.impl;

import searchclient.mcts.Node;
import searchclient.mcts.backpropagation.Backpropagation;

public class AdditiveBackpropagation implements Backpropagation {
    private static final int WIN_SCORE = 10;

    @Override
    public void backpropagate(int score, Node nodeToExplore, Node root) {
        Node tempNode = nodeToExplore;
        while (tempNode != null && !tempNode.equals(root)) {
            tempNode.incrementVisitCount();
            tempNode.addScore(score);
            tempNode = tempNode.getParent();
        }
    }
}
