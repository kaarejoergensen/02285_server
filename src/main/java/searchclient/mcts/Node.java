package searchclient.mcts;

import lombok.Data;
import searchclient.State;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

@Data
public class Node {
    private int visitCount = 0;
    private int winScore = 0;
    private int countToRoot;
    final private State state;
    private List<Node> children = new ArrayList<>();
    private Node parent;

    public Node(State state, Node parent) {
        this.state = state;
        this.countToRoot = parent.getCountToRoot() + 1;
        this.parent = parent;
    }

    public Node(State state) {
        this.state = state;
        this.countToRoot = 0;
        this.parent = null;
    }

    public Node getRandomChildNode() {
        int noOfPossibleMoves = this.children.size();
        int selectRandom = (int) (Math.random() * noOfPossibleMoves);
        return this.children.get(selectRandom);
    }

    public Node getLeafWithMaxScore() {
        if (!children.isEmpty()) {
            List<Node> nodes = new ArrayList<>();
            children.forEach(c -> nodes.add(c.getLeafWithMaxScore()));
            return Collections.max(nodes, Comparator.comparing(Node::getWinScore));
        }
        return this;
    }

    public Node getChildWithMaxScore() {
        return Collections.max(this.children, Comparator.comparing(Node::getWinScore));
    }

    public void addScore(int score) {
        if (this.winScore != Integer.MIN_VALUE)
            this.winScore += score;
    }

    public void incrementVisitCount() {
        this.visitCount++;
    }

    public Node makeRandomMove() {
        List<State> expandedStates = this.state.getExpandedStates();
        return expandedStates.size() > 0 ? new Node(expandedStates.get(0), this) : null;
    }
}