package searchclient.mcts.expansion.impl;

import searchclient.mcts.Node;
import searchclient.mcts.expansion.Expansion;

import java.util.List;
import java.util.stream.Collectors;

public class AllActionsExpansion implements Expansion {
    @Override
    public List<Node> expandNode(Node root) {
        List<Node> expandedStates = root.getState().getExpandedStates().stream().
                map(state -> new Node(state, root.getCountToRoot() + 1)).collect(Collectors.toList());
        root.getChildren().addAll(expandedStates);
        return expandedStates;
    }
}
