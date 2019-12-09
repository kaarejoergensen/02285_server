package searchclient.mcts.selection.impl;

import searchclient.mcts.model.Node;
import searchclient.mcts.selection.Selection;
import shared.Action;

public class AlphaGoSelection extends Selection {
    @Override
    public Node selectPromisingNode(Node rootNode) {
        Node node = rootNode;
        while (node.getChildren().size() != 0) {
            if (node.getState().isGoalState())
                return node;
            Action action = node.chooseBestAction();
            node = node.getActionChildMap().get(action);
        }
        return node;
    }

    @Override
    public int compare(Node o1, Node o2) {
        return 0;
    }
}
