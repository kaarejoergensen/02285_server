package searchclient.mcts.search.impl;

import searchclient.mcts.backpropagation.Backpropagation;
import searchclient.mcts.backpropagation.impl.AdditiveRAVEBackpropagation;
import searchclient.mcts.expansion.Expansion;
import searchclient.mcts.model.Node;
import searchclient.mcts.search.MonteCarloTreeSearch;
import searchclient.mcts.selection.Selection;
import searchclient.nn.NNet;
import searchclient.nn.PredictResult;
import shared.Action;

import java.util.*;

public class AlphaGo extends MonteCarloTreeSearch {
    private static final int MCTS_LOOP_ITERATIONS = 120;

    private NNet nNet;

    public AlphaGo(Selection selection, Expansion expansion, Backpropagation backpropagation,
                   NNet nNet) {
        super(selection, expansion, null, backpropagation);
        this.nNet = nNet;
    }

    @Override
    public Action[][] solve(Node root, boolean limitSolveTries) {
        Node node = root;
        Action[][] solution = null;
        for (int i = 0; (!limitSolveTries || i < SOLVE_TRIES) && solution == null; i++) {
            node = this.runMCTS(node).getChildWithMaxScore();
            if (node.getState().isGoalState()) {
                solution = node.getState().extractPlan();
            }
        }
	    return solution;
    }

    @Override
    public Node runMCTS(Node root) {
        if (this.backpropagation instanceof AdditiveRAVEBackpropagation) {
            ((AdditiveRAVEBackpropagation) this.backpropagation).addExpandedNodes(Collections.singletonList(root));
        }
        for (int i = 0; i < MCTS_LOOP_ITERATIONS; i++) {
            Node promisingNode = this.selection.selectPromisingNode(root);

            List<Node> expandedNodes = this.expansion.expandNode(promisingNode);
            if (this.backpropagation instanceof AdditiveRAVEBackpropagation) {
                ((AdditiveRAVEBackpropagation) this.backpropagation).addExpandedNodes(expandedNodes);
            }

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
