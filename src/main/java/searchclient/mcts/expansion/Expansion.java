package searchclient.mcts.expansion;

import searchclient.State;
import searchclient.mcts.model.Node;

import java.util.Collection;
import java.util.List;

public abstract class Expansion implements Cloneable {
    public abstract List<Node> expandNode(Node root);

    public abstract Collection<State> getExpandedStates();

    @Override
    public abstract Expansion clone();

    @Override
    public abstract String toString();
}
