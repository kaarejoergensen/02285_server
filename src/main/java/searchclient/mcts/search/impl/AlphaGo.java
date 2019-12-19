package searchclient.mcts.search.impl;

import me.tongfei.progressbar.ProgressBar;
import me.tongfei.progressbar.ProgressBarBuilder;
import searchclient.mcts.backpropagation.Backpropagation;
import searchclient.mcts.expansion.Expansion;
import searchclient.mcts.model.Node;
import searchclient.mcts.search.MonteCarloTreeSearch;
import searchclient.mcts.selection.Selection;
import searchclient.nn.NNet;
import searchclient.nn.PredictResult;
import shared.Action;

import java.util.Collection;
import java.util.List;
import java.util.Optional;

public class AlphaGo extends MonteCarloTreeSearch {
    private static final int MCTS_LOOP_ITERATIONS = 400;
    private static final int SOLVE_TRIES = 100;

    private NNet nNet;

    public AlphaGo(Selection selection, Expansion expansion, Backpropagation backpropagation,
                   NNet nNet) {
        super(selection, expansion, null, backpropagation);
        this.nNet = nNet;
    }

    @Override
    public Action[][] solve(Node root) {
        Node node = root;
        ProgressBar progressBar = new ProgressBarBuilder()
                .setPrintStream(System.err)
                .setUpdateIntervalMillis(300)
                .setInitialMax(SOLVE_TRIES)
                .setTaskName("Iterations")
                .build();
        for (int i = 0; i < SOLVE_TRIES; i++) {
            node = this.runMCTS(node).getChildWithMaxScore();
            if (node.getState().isGoalState()) {
                progressBar.close();
                return node.getState().extractPlan();
            }
            Optional<Node> possibleGoalNode = this.extractGoalNodeIfPossible(node);
            if (possibleGoalNode.isPresent()) return possibleGoalNode.get().getState().extractPlan();
            progressBar.step();
        }
        progressBar.close();
	    return null;
    }

    @Override
    public Node runMCTS(Node root) {
        for (int i = 0; i < MCTS_LOOP_ITERATIONS; i++) {
            Node promisingNode = this.selection.selectPromisingNode(root);

            this.expansion.expandNode(promisingNode);

            PredictResult trainResult = this.nNet.predict(promisingNode.getState());
            this.setProbabilityMap(trainResult.getProbabilityVector(), promisingNode);

            this.backpropagation.backpropagate(trainResult.getScore(), promisingNode, root);
        }
        return root;
    }

    private void setProbabilityMap(double[] probabilityVector, Node node) {
        List<Action> actions = Action.getAllActions();
        for (int i = 0; i < actions.size(); i++) {
            if (node.getActionProbabilityMap().containsKey(actions.get(i))) {
                node.getActionProbabilityMap().put(actions.get(i), probabilityVector[i]);
            }
        }
    }

    private Optional<Node> extractGoalNodeIfPossible(Node root) {
        if (root.getState().isGoalState()) return Optional.of(root);
        if (!root.childrenEmpty()) {
            Node child = root.getChildWithMaxScore();
            if (child != null) return this.extractGoalNodeIfPossible(child);
        }
        return Optional.empty();
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
    public MonteCarloTreeSearch clone()  {
        return new AlphaGo(this.selection, this.expansion.clone(), this.backpropagation.clone(), this.nNet);
    }

    @Override
    public String toString() {
        return "AG_" + this.selection.toString() + "_" + this.expansion.toString() + "_" + this.backpropagation.toString() + "_" + this.nNet.toString();
    }
}
