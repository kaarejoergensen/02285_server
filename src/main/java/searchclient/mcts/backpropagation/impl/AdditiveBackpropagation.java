package searchclient.mcts.backpropagation.impl;

import searchclient.mcts.backpropagation.Backpropagation;
import searchclient.mcts.model.Node;

public class AdditiveBackpropagation implements Backpropagation {

    @Override
    public void backpropagate(float score, Node nodeToExplore, Node root) {
        Node tempNode = nodeToExplore;
        nodeToExplore.setSimulationScore(score);
        while (tempNode != null) {
            tempNode.incrementVisitCount();
            tempNode.addScore(score);
            tempNode = tempNode.getParent();
        }
    }
}
