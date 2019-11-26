package searchclient.mcts.expansion;

import searchclient.State;
import searchclient.mcts.model.Node;

import java.util.Collection;
import java.util.List;

public interface Expansion {
    List<Node> expandNode(Node root);

    Collection<State> getExpandedStates();
}
