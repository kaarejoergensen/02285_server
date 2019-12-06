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
import java.util.Collections;
import java.util.Comparator;
import java.util.concurrent.TimeUnit;

public class Basic extends MonteCarloTreeSearch {
    private static final int MCTS_LOOP_ITERATIONS = 10000;
    @Setter private NNet nNet;

    public Basic(Selection selection, Expansion expansion, Simulation simulation, Backpropagation backpropagation) {
        super(selection, expansion, simulation, backpropagation);
    }

    public Node runMCTS(Node root, boolean train) {
        for (int i = 0; i < MCTS_LOOP_ITERATIONS; i++) {
            Node promisingNode = this.selection.selectPromisingNode(root);

//            if (!train && promisingNode.getState().isGoalState())
//                return promisingNode;

            this.expansion.expandNode(promisingNode);

            float score = this.simulation.simulatePlayout(promisingNode);
//            float score = train ? this.simulation.simulatePlayout(promisingNode) : this.nNet.predict(promisingNode.getState());

            this.backpropagation.backpropagate(score, promisingNode, root);
        }
        this.printPathToGoal(root);
        return train ? root : root.getChildWithMaxScore();
    }

    private void printPathToGoal(Node root) {
        Node node = root;
        while (node != null && !node.getChildren().isEmpty()) {
            System.out.println(node.getWinScore());
            System.out.println(node.getState());
            node = Collections.max(node.getChildren(), Comparator.comparing(Node::getWinScore));
            try {
                TimeUnit.SECONDS.sleep(2);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    public Action[][] solve(Node root) {
        Node node = root;
        int iterations = 0;
        while (true) {
            System.out.println("Try nr... " + iterations++);
            node = this.runMCTS(node, false);
            if (node.getState().isGoalState()) {
                return node.getState().extractPlan();
            }
            if(iterations % 10 == 0){
                System.out.println("Hmm... det tar litt tid det her eller hva");
            }
        }
    }

    @Override
    public Collection getExpandedStates() {
        return this.expansion.getExpandedStates();
    }

}
