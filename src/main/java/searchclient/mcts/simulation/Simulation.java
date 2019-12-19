package searchclient.mcts.simulation;

import searchclient.mcts.model.Node;

public abstract class Simulation {
    public abstract float simulatePlayout(Node node);

    @Override
    public abstract String toString();
}
