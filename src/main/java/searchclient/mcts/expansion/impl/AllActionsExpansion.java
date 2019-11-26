package searchclient.mcts.expansion.impl;

import searchclient.State;
import searchclient.mcts.expansion.Expansion;
import searchclient.mcts.model.Node;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

public class AllActionsExpansion implements Expansion {
    List<State> expandedStates = new ArrayList<>();

    @Override
    public List<Node> expandNode(Node root) {
        List<State> expandedStates = root.getState().getExpandedStates();
        List<Node> expandedNodes = expandedStates.stream().
                map(state -> new Node(state, root)).collect(Collectors.toList());
        this.expandedStates.addAll(expandedStates);
        root.getChildren().addAll(expandedNodes);
        root.setExpanded(true);
        return expandedNodes;
    }

    @Override
    public Collection<State> getExpandedStates() {
        return this.expandedStates;
    }
}
