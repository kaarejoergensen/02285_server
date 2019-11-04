package searchclient.mcts.simulation.impl;

import searchclient.mcts.Node;
import searchclient.mcts.simulation.Simulation;

public class RandomSimulation implements Simulation {
    private static final int GOAL_SEARCH_LIMIT = 180;

    @Override
    public boolean simulatePlayout(Node node) {
        Node tempNode = node;
        int i = 0;
        while (!tempNode.getState().isGoalState() && i < GOAL_SEARCH_LIMIT) {
            tempNode = tempNode.makeRandomMove();
            i++;
        }
        return tempNode.getState().isGoalState();
    }
}
