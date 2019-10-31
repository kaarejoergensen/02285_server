package searchclient;

import java.util.*;

public class MonteCarloTreeSearch {
    private static final int WIN_SCORE = 10;

    private final int MCTS_LOOP_ITERATIONS = 1000;
    private final int GOAL_SEARCH_LIMIT = 1000;

    private Map<Integer, State> generatedStates = new HashMap<>();

    public State findNextMove(State state) {
        for (int i = 0; i < MCTS_LOOP_ITERATIONS; i++) {
            // Phase 1 - Selection
            State promisingNode = selectPromisingNode(state);
            // Phase 2 - Expansion
            if (!promisingNode.isGoalState())
                expandNode(promisingNode);

            // Phase 3 - Simulation
            State nodeToExplore = promisingNode;
            if (promisingNode.children.size() > 0) {
                nodeToExplore = promisingNode.getRandomChildState();
            }
            boolean solved = simulateRandomPayout(nodeToExplore);
            if (solved) System.err.println("SOLVED");
             // Phase 4 - Update
            backPropagation(nodeToExplore, solved);
        }

        return state.getChildWithMaxScore();
    }

    private State selectPromisingNode(State rootNode) {
        //System.err.println("Starting selectPromisingNode");
        State node = rootNode;
        while (node.children.size() != 0) {
            node = node.children.get(new Random().nextInt(node.children.size()));
        }
        //System.err.println("SelectPromisingNode done");
        return node;
    }

    private void expandNode(State root) {
        //System.err.println("Starting expandNode");
        List<State> expandedStates = root.getExpandedStates();
        //for (State state : expandedStates) {
        //    if (this.generatedStates.containsKey(state.hashCode())) {
        //        root.children.add(this.generatedStates.get(state.hashCode()));
        //    } else {
         //       root.children.add(state);
         //       this.generatedStates.put(state.hashCode(), state);
          //  }
       // }
        root.children.addAll(root.getExpandedStates());
        //System.err.println("ExpandNode done");
    }

    private void backPropagation(State nodeToExplore, boolean solved) {
       // System.err.println("Starting backPropagation");
        State tempNode = nodeToExplore;
        while (tempNode != null) {
            tempNode.incrementVisitCount();
            if (solved) tempNode.addScore(WIN_SCORE);
            tempNode = tempNode.parent;
        }
       // System.err.println("BackPropagation done");
    }

    private boolean simulateRandomPayout(State node) {
       // System.err.println("Starting simulateRandomPayout");
        State tempNode = node;
        int i = 0;
        while (!tempNode.isGoalState() && i < GOAL_SEARCH_LIMIT) {
            tempNode = tempNode.makeRandomMove();
            i++;
        }
        //System.err.println("SimulateRandomPayout done");
        return tempNode.isGoalState();
    }

}
