package searchclient.mcts;

import lombok.AllArgsConstructor;
import searchclient.Memory;
import searchclient.mcts.backpropagation.Backpropagation;
import searchclient.mcts.expansion.Expansion;
import searchclient.mcts.model.Node;
import searchclient.mcts.selection.Selection;
import searchclient.mcts.simulation.Simulation;
import shared.Action;

@AllArgsConstructor
public abstract class MonteCarloTreeSearch {
    final protected Selection selection;
    final protected Expansion expansion;
    final protected Simulation simulation;
    final protected Backpropagation backpropagation;

    public abstract Action[][] solve(Node root);

    protected void printSearchStatus(long startTime, int expandedNodes, int totalIterations) {
        String statusTemplate = "#Expanded: %,8d, #Total iterations: %,8d, Time: %3.3f s\n%s\n";
        double elapsedTime = (System.nanoTime() - startTime) / 1_000_000_000d;
        System.err.format(statusTemplate, expandedNodes, totalIterations, elapsedTime, Memory.stringRep());
    }
}
