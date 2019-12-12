package searchclient.mcts.search.impl;

import lombok.Setter;
import searchclient.mcts.backpropagation.Backpropagation;
import searchclient.mcts.expansion.Expansion;
import searchclient.mcts.model.Node;
import searchclient.mcts.search.MonteCarloTreeSearch;
import searchclient.mcts.selection.Selection;
import searchclient.mcts.simulation.Simulation;
import searchclient.nn.NNet;
import shared.Action;

import java.util.Collection;

public class Basic extends MonteCarloTreeSearch {
    private static final int MCTS_LOOP_ITERATIONS = 1600;
    private NNet nNet;
    private boolean train = false;

    public Basic(Selection selection, Expansion expansion, Simulation simulation, Backpropagation backpropagation,
                 NNet nNet) {
        super(selection, expansion, simulation, backpropagation);
        this.nNet = nNet;
    }

    @Override
    public Action[][] solve(Node root) {
        Node node = root;
        int iterations = 0;
        while (true) {
            System.out.println("Try nr... " + iterations++);
            node = this.runMCTS(node);
            if (node.getState().isGoalState()) {
                return node.getState().extractPlan();
            }
            if(iterations % 10 == 0){
                System.out.println("Hmm... det tar litt tid det her eller hva");
            }
        }
    }

    @Override
    public Node runMCTS(Node root) {
        for (int i = 0; i < MCTS_LOOP_ITERATIONS; i++) {
            Node promisingNode = this.selection.selectPromisingNode(root);

            if (!train && promisingNode.getState().isGoalState())
                return promisingNode;

            this.expansion.expandNode(promisingNode);

            float score = train ? this.simulation.simulatePlayout(promisingNode) : this.nNet.predict(promisingNode.getState()).getScore();

            this.backpropagation.backpropagate(score, promisingNode, root);
        }
        return train ? root : root.getChildWithMaxScore();
    }

    @Override
    public Collection<?> getExpandedStates() {
        return this.expansion.getExpandedStates();
    }

    @Override
    public void setNNet(NNet nNet) {
        this.nNet = nNet;
    }

    @Override
    public MonteCarloTreeSearch clone() {
        return new Basic(this.selection, this.expansion.clone(), this.simulation, this.backpropagation, this.nNet);
    }

}
