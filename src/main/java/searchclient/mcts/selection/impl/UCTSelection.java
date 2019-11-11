package searchclient.mcts.selection.impl;

import searchclient.mcts.model.Node;
import searchclient.mcts.selection.Selection;

import java.util.Collections;
import java.util.Comparator;

public class UCTSelection extends Selection {
    @Override
    public Node selectPromisingNode(Node rootNode) {
        Node node = rootNode;
        while (node.getChildren().size() != 0) {
            node = this.findBestNodeWithUCT(node);
        }
        return node;
    }

    private Node findBestNodeWithUCT(Node node) {
        return Collections.max(node.getChildren(), Comparator.comparing(this::uctValue));
    }

    private double uctValue(Node node) {
        int parentVisit = node.getParent() == null ? 0 : node.getParent().getVisitCount();
        if (parentVisit == 0) return Integer.MAX_VALUE;
        return (node.getWinScore() / (double) parentVisit) + 1.41 * Math.sqrt(Math.log(parentVisit) / (double) node.getVisitCount());
    }

    @Override
    public int compare(Node node1, Node node2) {
        return Math.negateExact((int) (this.uctValue(node1) - this.uctValue(node2)));
    }
}
