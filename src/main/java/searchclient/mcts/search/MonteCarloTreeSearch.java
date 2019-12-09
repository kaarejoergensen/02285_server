package searchclient.mcts.search;

import lombok.AllArgsConstructor;
import searchclient.Memory;
import searchclient.mcts.backpropagation.Backpropagation;
import searchclient.mcts.expansion.Expansion;
import searchclient.mcts.model.Node;
import searchclient.mcts.selection.Selection;
import searchclient.mcts.simulation.Simulation;
import shared.Action;

import java.util.Collection;

@AllArgsConstructor
public abstract class MonteCarloTreeSearch {
    final protected Selection selection;
    final protected Expansion expansion;
    final protected Simulation simulation;
    final protected Backpropagation backpropagation;

    public abstract Action[][] solve(Node root);

    public abstract Collection<?> getExpandedStates();
}
