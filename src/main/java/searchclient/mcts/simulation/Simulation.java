package searchclient.mcts.simulation;

import searchclient.mcts.model.Node;

public interface Simulation {
    float simulatePlayout(Node node);
}
