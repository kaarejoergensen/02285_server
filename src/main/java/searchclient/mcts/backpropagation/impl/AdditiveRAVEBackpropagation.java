package searchclient.mcts.backpropagation.impl;

import searchclient.State;
import searchclient.mcts.backpropagation.Backpropagation;
import searchclient.mcts.model.Node;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class AdditiveRAVEBackpropagation extends AdditiveBackpropagation {
    private Map<State, Set<Node>> nodeMap = new HashMap<>();
    private boolean concurrentEnabled = false;

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
                if (!concurrentEnabled) nodeList = new HashSet<>();
                else nodeList = ConcurrentHashMap.newKeySet();
            }
            nodeList.add(node);
            this.nodeMap.put(node.getState(), nodeList);
        }
    }

    public void setConcurrentEnabled() {
        this.nodeMap = new ConcurrentHashMap<>();
        this.concurrentEnabled = true;
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
