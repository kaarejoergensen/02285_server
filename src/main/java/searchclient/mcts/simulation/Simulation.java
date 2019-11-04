package searchclient.mcts.simulation;

import searchclient.mcts.Node;

public interface Simulation {
    boolean simulatePlayout(Node node);
}
