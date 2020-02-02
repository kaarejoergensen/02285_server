package searchclient.mcts.search.impl;

import searchclient.NotImplementedException;
import searchclient.mcts.backpropagation.Backpropagation;
import searchclient.mcts.expansion.Expansion;
import searchclient.mcts.expansion.impl.AllActionsNoDuplicatesExpansion;
import searchclient.mcts.model.Node;
import searchclient.mcts.search.MonteCarloTreeSearch;
import searchclient.mcts.selection.Selection;
import searchclient.mcts.simulation.Simulation;
import shared.Action;

import java.util.Collection;
import java.util.List;
import java.util.PriorityQueue;

public class OneTree extends MonteCarloTreeSearch {
    private PriorityQueue<Node> frontier;
    private static final int searchDepth = 32;
    private int currentDepth = 0;

    public OneTree(Selection selection, Expansion expansion, Simulation simulation, Backpropagation backpropagation) {
        super(selection, expansion, simulation, backpropagation);
        this.frontier = new PriorityQueue<>(this.selection);
    }

    @Override
    public Action[][] solve(Node root, boolean limitSolveTries) {
        Action[][] solution = null;
        for (int i = 0; (!limitSolveTries || i < SOLVE_TRIES) && solution == null; i++) {
            Node promisingNode = this.selection.selectPromisingNode(root);
            List<Node> expandedNodes = this.expansion.expandNode(promisingNode);
            if (expandedNodes.isEmpty()) {
                if (promisingNode.getParent() != null) promisingNode.getParent().removeChild(promisingNode);
                if (promisingNode.equals(root)) {
                    root = this.nextRoot(root);
                }
                continue;
            }
            for (Node node : expandedNodes) {
                if (node.getState().isGoalState()) {
                    solution = node.getState().extractPlan();
                }

                float score = this.simulation.simulatePlayout(node);
                this.backpropagation.backpropagate(score, node, root);

                if (this.currentDepth < node.getCountToRoot()) {
                    this.currentDepth = node.getCountToRoot();
                }
            }

            if (currentDepth >= (searchDepth + root.getCountToRoot())) {
                root = this.nextRoot(root);
            }
        }
        return solution;
    }

    @Override
    public Node runMCTS(Node root) {
        throw new NotImplementedException();
    }

    @Override
    public Collection<?> getExpandedStates() {
        return this.expansion.getExpandedStates();
    }

    @Override
    public MonteCarloTreeSearch clone() {
        return new OneTree(this.selection, this.expansion.clone(), this.simulation, this.backpropagation.clone());
    }

    @Override
    public String toString() {
        return "OT_" + this.selection.toString() + "_" + this.expansion.toString() + "_" + this.backpropagation.toString();
    }

    private Node nextRoot(Node root) {
        this.currentDepth = 0;
        this.frontier.addAll(root.getChildren());
        return this.frontier.poll();
    }
}
