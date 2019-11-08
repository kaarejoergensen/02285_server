package searchclient.mcts.simulation;

import searchclient.mcts.Node;

public interface Simulation {
    int simulatePlayout(Node node);
}
