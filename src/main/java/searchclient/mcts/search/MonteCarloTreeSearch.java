package searchclient.mcts.search;

import lombok.AllArgsConstructor;
import searchclient.NotImplementedException;
import searchclient.mcts.backpropagation.Backpropagation;
import searchclient.mcts.expansion.Expansion;
import searchclient.mcts.model.Node;
import searchclient.mcts.selection.Selection;
import searchclient.mcts.simulation.Simulation;
import searchclient.nn.NNet;
import shared.Action;

import java.util.Collection;

@AllArgsConstructor
public abstract class MonteCarloTreeSearch implements Cloneable {
    protected static final int SOLVE_TRIES = 100;

    final protected Selection selection;
    final protected Expansion expansion;
    final protected Simulation simulation;
    final protected Backpropagation backpropagation;

    public abstract Action[][] solve(Node root, boolean limitSolveTries);

    public abstract Node runMCTS(Node root);

    public abstract Collection<?> getExpandedStates();

    public void setNNet(NNet nNet) {
        throw new NotImplementedException();
    }

    @Override
    public abstract MonteCarloTreeSearch clone();

    @Override
    public abstract String toString();
}
