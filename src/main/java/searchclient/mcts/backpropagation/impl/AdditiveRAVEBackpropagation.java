package searchclient.mcts.backpropagation.impl;

import lombok.AllArgsConstructor;
import lombok.Data;
import searchclient.State;
import searchclient.mcts.backpropagation.Backpropagation;
import searchclient.mcts.model.Node;
import shared.Action;

import java.util.*;

public class AdditiveRAVEBackpropagation extends Backpropagation {
    Map<StateActionPair, List<Node>> nodeMap = new HashMap<>();

    @Override
    public void backpropagate(float score, Node nodeToExplore, Node root) {
        Node tempNode = nodeToExplore;
        while (tempNode != null && tempNode.getParent() != null) {
            Action actionPerformed = tempNode.getActionPerformed();
            StateActionPair stateActionPair = new StateActionPair(tempNode.getState(), actionPerformed);
            List<Node> nodeList;
            if (this.nodeMap.containsKey(stateActionPair)) {
                nodeList = this.nodeMap.get(stateActionPair);
            } else {
                nodeList = new ArrayList<>();
            }
            nodeList.add(tempNode);
            for (Node node : nodeList) {
                Node parent = node.getParent();
                parent.incrementVisitCount(actionPerformed);
                parent.addScore(actionPerformed, score);
            }
            tempNode = tempNode.getParent();
            this.nodeMap.put(stateActionPair, nodeList);
        }
    }

    @Override
    public Backpropagation clone() {
        return new AdditiveRAVEBackpropagation();
    }

    @Data
    @AllArgsConstructor
    private static class StateActionPair {
        private State state;
        private Action action;
    }
}
