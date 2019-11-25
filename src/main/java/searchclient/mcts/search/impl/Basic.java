package searchclient.mcts.search.impl;

import searchclient.mcts.backpropagation.Backpropagation;
import searchclient.mcts.expansion.Expansion;
import searchclient.mcts.model.Node;
import searchclient.mcts.search.MonteCarloTreeSearch;
import searchclient.mcts.selection.Selection;
import searchclient.mcts.simulation.Simulation;
import shared.Action;

import java.util.*;

public class Basic extends MonteCarloTreeSearch {
    private static final int MCTS_LOOP_ITERATIONS = 1000;

    public Basic(Selection selection, Expansion expansion, Simulation simulation, Backpropagation backpropagation) {
        super(selection, expansion, simulation, backpropagation);
    }

    private Node findNextMove(Node root) {
        for (int i = 0; i < MCTS_LOOP_ITERATIONS; i++) {
            Node promisingNode = this.selection.selectPromisingNode(root);

            if (promisingNode.getState().isGoalState())
                return promisingNode;

            this.expansion.expandNode(promisingNode);

            float score = this.simulation.simulatePlayout(promisingNode);

            this.backpropagation.backpropagate(score, promisingNode, root);
        }

        return root.getChildWithMaxScore();
    }

    @Override
    public Action[][] solve(Node root) {
        Node node = root;
        while (true) {
            node = this.findNextMove(node);
            if (node.getState().isGoalState()) {
                return node.getState().extractPlan();
            }
        }
    }

    @Override
    public Collection getExpandedStates() {
        return this.expansion.getExpandedStates();
    }

}
