package searchclient.mcts.search.impl;

import searchclient.mcts.backpropagation.Backpropagation;
import searchclient.mcts.expansion.Expansion;
import searchclient.mcts.model.Node;
import searchclient.mcts.search.MonteCarloTreeSearch;
import searchclient.mcts.selection.Selection;
import searchclient.nn.NNet;
import searchclient.nn.PredictResult;
import shared.Action;

import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Optional;

public class AlphaGo extends MonteCarloTreeSearch {
    private static final int MCTS_LOOP_ITERATIONS = 120;
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
        Action[][] solution = null;
        int i;
        for (i = 0; i < SOLVE_TRIES && solution == null; i++) {
            node = this.runMCTS(node).getChildWithMaxScore();
            if (node.getState().isGoalState()) {
                solution = node.getState().extractPlan();
            }
//            Optional<Node> possibleGoalNode = this.extractGoalNodeIfPossible(node);
//            if (possibleGoalNode.isPresent()) solution = possibleGoalNode.get().getState().extractPlan();
        }
        if (solution == null)
            System.err.println("No solution found in " + SOLVE_TRIES + " iterations.");
        else
            System.err.println("Solution found in " + i + " iterations.");
	    return solution;
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
        if (probabilityVector.length < actions.size()) {
            System.err.println("SHORTER!");
            System.err.println(Arrays.toString(probabilityVector));
        }
        for (int i = 0; i < actions.size(); i++) {
            if (node.getActionProbabilityMap().containsKey(actions.get(i))) {
                node.getActionProbabilityMap().put(actions.get(i), probabilityVector[i]);
            }
        }
    }

    private Optional<Node> extractGoalNodeIfPossible(Node root) {
        if (root.getState().isGoalState()) return Optional.of(root);
        if (!root.childrenEmpty()) {
            for (Node child : root.getChildren()) {
                Optional<Node> optionalNode = this.extractGoalNodeIfPossible(child);
                if (optionalNode.isPresent()) return optionalNode;
            }
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
