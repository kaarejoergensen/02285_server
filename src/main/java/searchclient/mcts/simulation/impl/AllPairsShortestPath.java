package searchclient.mcts.simulation.impl;

import searchclient.Heuristic;
import searchclient.State;
import searchclient.mcts.model.Node;
import searchclient.mcts.simulation.Simulation;

public class AllPairsShortestPath implements Simulation {
    private static final float MAX_SCORE = 100;

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
            return MAX_SCORE;
        }

//        System.err.println(result + "|" + MAX_SCORE / result);
        return MAX_SCORE / (float) this.heuristic.h(n);
    }
}
