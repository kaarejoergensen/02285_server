package searchclient.mcts.search.impl;

import searchclient.mcts.backpropagation.Backpropagation;
import searchclient.mcts.expansion.Expansion;
import searchclient.mcts.model.Node;
import searchclient.mcts.search.MonteCarloTreeSearch;
import searchclient.mcts.selection.Selection;
import searchclient.mcts.simulation.Simulation;
import shared.Action;

import java.util.Collection;

public class Basic extends MonteCarloTreeSearch {
    private static final int MCTS_LOOP_ITERATIONS = 400;
    private static final int SOLVE_TRIES = 10000;

    public Basic(Selection selection, Expansion expansion, Simulation simulation, Backpropagation backpropagation) {
        super(selection, expansion, simulation, backpropagation);
    }

    @Override
    public Action[][] solve(Node root) {
        Node node = root;
        Action[][] solution = null;
        int i;
        for (i = 0; i < SOLVE_TRIES && solution == null; i++) {
            node = this.runMCTS(node);
            if (node.getState().isGoalState()) {
                solution = node.getState().extractPlan();
            }
        }
        if (solution == null)
            System.err.println("No solution found in " + SOLVE_TRIES + " iterations.");
        else
            System.err.println("Solution found in " + i + " iterations.");
        return solution;
    }

    @Override
    public Node runMCTS(Node root) {
        for (int i = 0; i < MCTS_LOOP_ITERATIONS; i++) {
            Node promisingNode = this.selection.selectPromisingNode(root);

            if (promisingNode.getState().isGoalState())
                return promisingNode;

            this.expansion.expandNode(promisingNode);

            float score =  this.simulation.simulatePlayout(promisingNode);

            this.backpropagation.backpropagate(score, promisingNode, root);
        }
        return root.getChildWithMaxScore();
    }

    @Override
    public Collection<?> getExpandedStates() {
        return this.expansion.getExpandedStates();
    }

    @Override
    public MonteCarloTreeSearch clone() {
        return new Basic(this.selection, this.expansion.clone(), this.simulation, this.backpropagation.clone());
    }

    @Override
    public String toString() {
        return "Basic_" + this.selection.toString() + "_" + this.expansion.toString() + "_" + this.simulation.toString() + "_" + this.backpropagation.toString();
    }

}
