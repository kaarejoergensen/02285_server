package searchclient.mcts;

import lombok.Data;
import searchclient.State;
import searchclient.mcts.backpropagation.Backpropagation;
import searchclient.mcts.expansion.Expansion;
import searchclient.mcts.selection.Selection;
import searchclient.mcts.simulation.Simulation;

import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

@Data
public class MonteCarloTreeSearch {
    private static final int MCTS_LOOP_ITERATIONS = 100000;

    final private Selection selection;
    final private Expansion expansion;
    final private Simulation simulation;
    final private Backpropagation backpropagation;

    Set<Node> expandedNodes = new HashSet<>();

    public State findNextMove(Node root) {
        for (int i = 0; i < MCTS_LOOP_ITERATIONS; i++) {
            Node promisingNode = this.selection.selectPromisingNode(root);

            if (!promisingNode.getState().isGoalState())
                this.expandedNodes.addAll(this.expansion.expandNode(promisingNode));

            int score = this.simulation.simulatePlayout(promisingNode);

            this.backpropagation.backpropagate(score, promisingNode, root);
        }

        return root.getChildWithMaxScore().getState();
    }
}
