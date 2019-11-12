package searchclient.mcts.search.impl;

import searchclient.mcts.backpropagation.Backpropagation;
import searchclient.mcts.expansion.Expansion;
import searchclient.mcts.model.Node;
import searchclient.mcts.search.MonteCarloTreeSearch;
import searchclient.mcts.selection.Selection;
import searchclient.mcts.simulation.Simulation;
import shared.Action;

import java.util.HashSet;
import java.util.Set;

public class Basic extends MonteCarloTreeSearch {
    private static final int MCTS_LOOP_ITERATIONS = 1000;

    private Set<Node> expandedNodes = new HashSet<>();

    public Basic(Selection selection, Expansion expansion, Simulation simulation, Backpropagation backpropagation) {
        super(selection, expansion, simulation, backpropagation);
    }

    private Node findNextMove(Node root) {
        for (int i = 0; i < MCTS_LOOP_ITERATIONS; i++) {
            Node promisingNode = this.selection.selectPromisingNode(root);

            if (!promisingNode.getState().isGoalState())
                this.expandedNodes.addAll(this.expansion.expandNode(promisingNode));

            float score = this.simulation.simulatePlayout(promisingNode);

            this.backpropagation.backpropagate(score, promisingNode, root);
        }

        return root.getChildWithMaxScore();
    }

    @Override
    public Action[][] solve(Node root) {
        long startTime = System.nanoTime();
        int iterations = 0;
        Node node = root;
        while (true) {
            node = this.findNextMove(node);
            printSearchStatus(startTime, this.expandedNodes.size(), (iterations++) * MCTS_LOOP_ITERATIONS);
            if (node.getState().isGoalState()) {
                return node.getState().extractPlan();
            }
        }
    }
}
