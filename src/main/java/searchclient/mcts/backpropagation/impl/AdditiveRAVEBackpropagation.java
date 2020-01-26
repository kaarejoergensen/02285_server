package searchclient.mcts.backpropagation.impl;

import searchclient.State;
import searchclient.mcts.backpropagation.Backpropagation;
import searchclient.mcts.model.Node;

import java.util.*;

public class AdditiveRAVEBackpropagation extends AdditiveBackpropagation {
    private Map<State, Set<Node>> nodeMap = new HashMap<>();

    @Override
    public void backpropagate(float score, Node nodeToExplore, Node root) {
        Set<Node> nodes = this.nodeMap.get(nodeToExplore.getState());
        if (nodes == null) {
            throw new IllegalArgumentException("List of Nodes from nodeMap cannot be null!");
        }
        for (Node node : nodes) {
            super.backpropagate(score, node, root);
        }
    }

    public void addExpandedNodes(List<Node> expandedNodes) {
        for (Node node : expandedNodes) {
            Set<Node> nodeList;
            if (this.nodeMap.containsKey(node.getState())) {
                nodeList = this.nodeMap.get(node.getState());
            } else {
                nodeList = new HashSet<>();
            }
            nodeList.add(node);
            this.nodeMap.put(node.getState(), nodeList);
        }
    }

    @Override
    public Backpropagation clone() {
        return new AdditiveRAVEBackpropagation();
    }

    @Override
    public String toString() {
        return "ARB";
    }
}
