package searchclient.mcts.backpropagation.impl;

import searchclient.mcts.backpropagation.Backpropagation;
import searchclient.mcts.model.Node;
import shared.Action;

public class AdditiveBackpropagation extends Backpropagation {

    @Override
    public void backpropagate(float score, Node nodeToExplore, Node root) {
        Node tempNode = nodeToExplore;
        while (tempNode != null && tempNode.getParent() != null) {
            Action actionPerformed = tempNode.getActionPerformed();
            Node parent = tempNode.getParent();
            parent.incrementVisitCount(actionPerformed);
            parent.addScore(actionPerformed, score);
            tempNode = parent;
        }
    }

    @Override
    public Backpropagation clone() {
        return new AdditiveRAVEBackpropagation();
    }
}
