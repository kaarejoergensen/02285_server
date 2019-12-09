package searchclient.mcts.search.impl;

import searchclient.mcts.backpropagation.Backpropagation;
import searchclient.mcts.expansion.Expansion;
import searchclient.mcts.model.Node;
import searchclient.mcts.search.MonteCarloTreeSearch;
import searchclient.mcts.selection.Selection;
import searchclient.mcts.simulation.Simulation;
import searchclient.nn.NNet;
import shared.Action;

import java.util.Collection;

public class AlphaGo extends MonteCarloTreeSearch {
    private static final int MCTS_LOOP_ITERATIONS = 1600;
    private NNet nNet;
    private boolean train;

    public AlphaGo(Selection selection, Expansion expansion, Simulation simulation, Backpropagation backpropagation,
                   NNet nNet, boolean train) {
        super(selection, expansion, simulation, backpropagation);
        this.nNet = nNet;
        this.train = train;
    }

    private Node runMCTS(Node root) {
        for (int i = 0; i < MCTS_LOOP_ITERATIONS; i++) {
            Node promisingNode = this.selection.selectPromisingNode(root);

            this.expansion.expandNode(promisingNode);

            //TODO: Implement score and probability map
            float score = this.nNet.predict(promisingNode.getState());

            this.backpropagation.backpropagate(score, promisingNode, root);
        }
        return this.train ? root.getChildStochastic(true) : root.getChildWithMaxScore();
    }

    @Override
    public Action[][] solve(Node root) {
        Node node = root;
        while (true) {
            node = this.runMCTS(node);
            if (node.getState().isGoalState()) {
                return node.getState().extractPlan();
            }
        }
    }

    @Override
    public Collection<?> getExpandedStates() {
        return this.expansion.getExpandedStates();
    }
}
