package searchclient.mcts.simulation.impl;

import searchclient.Heuristic;
import searchclient.State;
import searchclient.mcts.model.Node;
import searchclient.mcts.simulation.Simulation;

public class AllPairsShortestPath extends Simulation {
    private Heuristic heuristic;

    public AllPairsShortestPath(State initialState) {
        this.heuristic = new Heuristic(initialState) {
            @Override
            public int f(State n) {
                return this.h(n);
            }
        };
    }

    @Override
    public float simulatePlayout(Node node) {
        State n = node.getState();
        if (n.isGoalState()) {
            return 1f;
        }

        return (float) (1f / Math.sqrt(this.heuristic.h(n)));
    }

    @Override
    public String toString() {
        return "APSPS";
    }
}
