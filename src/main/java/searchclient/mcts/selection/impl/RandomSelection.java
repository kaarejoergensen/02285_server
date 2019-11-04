package searchclient.mcts.selection.impl;

import searchclient.mcts.Node;
import searchclient.mcts.selection.Selection;

import java.util.Random;

public class RandomSelection implements Selection {
    @Override
    public Node selectPromisingNode(Node rootNode) {
        Node node = rootNode;
        while (node.getChildren().size() != 0) {
            node = node.getChildren().get(new Random().nextInt(node.getChildren().size()));
        }
        return node;
    }
}
