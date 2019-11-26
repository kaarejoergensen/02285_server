package searchclient.mcts.selection.impl;

import searchclient.State;
import searchclient.mcts.model.Node;
import searchclient.mcts.selection.Selection;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;

public class RandomSelection extends Selection {
    private Random random = new Random();
    @Override
    public Node selectPromisingNode(Node rootNode) {
        Node node = rootNode;
        while (!node.getChildren().isEmpty() && !node.isExpanded()) {
            if (node.getChildren().isEmpty() && node.isExpanded()) node = rootNode;
            node = node.getChildren().get(this.random.nextInt(node.getChildren().size()));
        }
        return node;
    }

    @Override
    public int compare(Node o1, Node o2) {
        return random.nextInt(3) - 1;
    }
}
