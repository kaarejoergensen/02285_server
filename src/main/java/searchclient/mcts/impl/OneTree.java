package searchclient.mcts.impl;

import searchclient.mcts.MonteCarloTreeSearch;
import searchclient.mcts.backpropagation.Backpropagation;
import searchclient.mcts.expansion.Expansion;
import searchclient.mcts.expansion.impl.AllActionsNoDuplicateExpansion;
import searchclient.mcts.model.Node;
import searchclient.mcts.selection.Selection;
import searchclient.mcts.simulation.Simulation;
import shared.Action;

import java.util.List;
import java.util.PriorityQueue;

public class OneTree extends MonteCarloTreeSearch {
    private PriorityQueue<Node> frontier;
    private int searchDepth = 32;
    private int currentDepth = 0;

    public OneTree(Selection selection, Expansion expansion, Simulation simulation, Backpropagation backpropagation) {
        super(selection, expansion, simulation, backpropagation);
        this.frontier = new PriorityQueue<>(this.selection);
    }

    @Override
    public Action[][] solve(Node root) {
        long startTime = System.nanoTime();
        int iterations = 0;
        int totalIterations = 0;
        List<Node> expandedNodes;

        while (true) {
            if (iterations == 10000) {
                printSearchStatus(startTime, ((AllActionsNoDuplicateExpansion) this.expansion).getExpandedStates().size(), totalIterations);
                iterations = 0;
            }
            Node promisingNode = this.selection.selectPromisingNode(root);
            expandedNodes = this.expansion.expandNode(promisingNode);
            if (expandedNodes.isEmpty()) {
                if (promisingNode.getParent() != null) promisingNode.getParent().removeChild(promisingNode);
                if (promisingNode.equals(root)) {
                    root = this.nextRoot(root);
                }
                continue;
            }
            for (Node node : expandedNodes) {
                if (node.getState().isGoalState()) return node.getState().extractPlan();
                float score = this.simulation.simulatePlayout(promisingNode);
                this.backpropagation.backpropagate(score, promisingNode, root);

                if (this.currentDepth < node.getCountToRoot()) {
                    this.currentDepth = node.getCountToRoot();
                }
            }

            if (currentDepth >= (searchDepth + root.getCountToRoot())) {
                root = this.nextRoot(root);
            }
            iterations++;
            totalIterations++;
        }
    }

    private Node nextRoot(Node root) {
        this.currentDepth = 0;
        this.frontier.addAll(root.getChildren());
        return this.frontier.poll();
    }
}
