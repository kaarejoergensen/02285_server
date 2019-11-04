package searchclient.mcts.selection.impl;

import searchclient.mcts.Node;
import searchclient.mcts.selection.Selection;

import java.util.Collections;
import java.util.Comparator;

public class UCTSelection implements Selection {
    @Override
    public Node selectPromisingNode(Node rootNode) {
        Node node = rootNode;
        while (node.getChildren().size() != 0) {
            node = this.findBestNodeWithUCT(node);
        }
        return node;
    }

    private Node findBestNodeWithUCT(Node node) {
        int parentVisit = node.getVisitCount();
        return Collections.max(
                node.getChildren(),
                Comparator.comparing(c -> this.uctValue(parentVisit, c.getWinScore(), c.getVisitCount())));
    }

    private double uctValue(int totalVisit, double nodeWinScore, int nodeVisit) {
        if (nodeVisit == 0) {
            return Integer.MAX_VALUE;
        }
        return (nodeWinScore / (double) nodeVisit) + 1.41 * Math.sqrt(Math.log(totalVisit) / (double) nodeVisit);
    }
}
