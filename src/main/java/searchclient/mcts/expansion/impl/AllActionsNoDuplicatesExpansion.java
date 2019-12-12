package searchclient.mcts.expansion.impl;

import lombok.RequiredArgsConstructor;
import searchclient.State;
import searchclient.mcts.expansion.Expansion;
import searchclient.mcts.model.Node;

import java.util.*;

@RequiredArgsConstructor
public class AllActionsNoDuplicatesExpansion implements Expansion {
    private final State root;
    private HashMap<State, Node> expandedStates = new HashMap<>();

    @Override
    public List<Node> expandNode(Node root) {
        List<State> newExpandedStates = root.getState().getExpandedStates();
        List<Node> nodes = new ArrayList<>();
        for (State state : newExpandedStates) {
            if (state.equals(this.root)) continue;
            if (!this.expandedStates.containsKey(state)) {
                Node node = new Node(state, root, state.jointAction[0]);
                this.expandedStates.put(state, node);
                nodes.add(new Node(state, root, state.jointAction[0]));
            } else {
                nodes.add(this.expandedStates.get(state));
            }
        }
        root.addChildren(nodes);
        root.setExpanded(true);
        return nodes;
    }

    @Override
    public Collection<State> getExpandedStates() {
        return this.expandedStates.keySet();
    }
}
