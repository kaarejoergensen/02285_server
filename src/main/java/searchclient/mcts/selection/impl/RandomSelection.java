package searchclient.mcts.selection.impl;

import searchclient.mcts.Node;
import searchclient.mcts.selection.Selection;

import java.util.Random;

public class RandomSelection extends Selection {
    Random random = new Random();
    @Override
    public Node selectPromisingNode(Node rootNode) {
        Node node = rootNode;
        while (node.getChildren().size() != 0) {
            node = node.getChildren().get(this.random.nextInt(node.getChildren().size()));
        }
        return node;
    }

    @Override
    public int compare(Node o1, Node o2) {
        return random.nextInt(3) - 1;
    }
}
