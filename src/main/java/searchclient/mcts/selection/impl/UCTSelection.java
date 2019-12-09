package searchclient.mcts.selection.impl;

import lombok.RequiredArgsConstructor;
import searchclient.mcts.model.Node;
import searchclient.mcts.selection.Selection;

import java.util.Collections;
import java.util.Comparator;
import java.util.Random;

@RequiredArgsConstructor
public class UCTSelection extends Selection {
    private final double constant;

    @Override
    public Node selectPromisingNode(Node rootNode) {
        Node node = rootNode;
        while (node.getChildren().size() != 0) {
            if (node.getState().isGoalState())
                return node;
            node = this.findBestNodeWithUCT(node);
        }
        return node;
    }

    private Node findBestNodeWithUCT(Node node) {
        return Collections.max(node.getChildren(), Comparator.comparing(this::uctValue));
    }

    private double uctValue(Node node) {
        //TODO: Fix this
//        int parentVisit = node.getParent() == null ? 0 : node.getParent().getVisitCount();
//        if (parentVisit == 0) return Integer.MAX_VALUE;
//        return (node.getTotalScore() / (double) parentVisit) + this.constant * Math.sqrt(Math.log(parentVisit) / (double) node.getVisitCount());
        return new Random().nextDouble();
    }

    @Override
    public int compare(Node node1, Node node2) {
        return Math.negateExact((int) (this.uctValue(node1) - this.uctValue(node2)));
    }
}
