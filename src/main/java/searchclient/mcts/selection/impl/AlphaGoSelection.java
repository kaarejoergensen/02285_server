package searchclient.mcts.selection.impl;

import searchclient.mcts.model.Node;
import searchclient.mcts.selection.Selection;
import shared.Action;

public class AlphaGoSelection extends Selection {
    @Override
    public Node selectPromisingNode(Node rootNode) {
        Node node = rootNode;
        while (!node.childrenEmpty()) {
            if (node.getState().isGoalState())
                return node;
            Action action = node.chooseBestAction();
            Node newNode = node.getActionChildMap().get(action);
            if (newNode == null) return node;
            node = newNode;
        }
        return node;
    }

    @Override
    public String toString() {
        return "AGS";
    }

    @Override
    public int compare(Node o1, Node o2) {
        return 0;
    }
}
