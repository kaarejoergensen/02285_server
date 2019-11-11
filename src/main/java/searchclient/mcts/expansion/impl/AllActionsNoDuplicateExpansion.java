package searchclient.mcts.expansion.impl;

import lombok.Getter;
import searchclient.State;
import searchclient.mcts.expansion.Expansion;
import searchclient.mcts.model.Node;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class AllActionsNoDuplicateExpansion implements Expansion {
    @Getter private Set<State> expandedStates = new HashSet<>();

    @Override
    public List<Node> expandNode(Node root) {
        List<State> newExpandedStates = root.getState().getExpandedStates();
        List<Node> nodes = new ArrayList<>();
        for (State state : newExpandedStates) {
            if (!this.expandedStates.contains(state)) {
                this.expandedStates.add(state);
                nodes.add(new Node(state, root));
            }
        }
        root.getChildren().addAll(nodes);

        return nodes;
    }
}
