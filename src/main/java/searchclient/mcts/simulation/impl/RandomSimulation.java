package searchclient.mcts.simulation.impl;

import searchclient.State;
import searchclient.mcts.model.Node;
import searchclient.mcts.simulation.Simulation;

import java.util.HashSet;
import java.util.Set;

public class RandomSimulation extends Simulation {
    private static final int GOAL_SEARCH_LIMIT = 180;

    @Override
    public float simulatePlayout(Node node) {
        Set<State> simulatedStates = new HashSet<>();
        Node tempNode = node;
        int i = 0;
        while (!tempNode.getState().isGoalState() && i < GOAL_SEARCH_LIMIT) {
            simulatedStates.add(tempNode.getState());
            Node newNode = tempNode.makeRandomMove(simulatedStates);
            tempNode = newNode != null ? newNode : tempNode.getParent();
            i++;
            if (tempNode == null) {
                throw new RuntimeException("No goal state could be found");
            }
        }
        return tempNode.getState().isGoalState() ? (float) (1f / Math.sqrt(tempNode.getCountToRoot())) : 0f;
    }

    @Override
    public String toString() {
        return "RS";
    }
}
