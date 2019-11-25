package searchclient.mcts.model;

import lombok.Getter;
import lombok.Setter;
import searchclient.State;

import java.util.*;

@Getter
@Setter
public class Node {
    private int visitCount = 0;
    private float winScore = 0;
    private int countToRoot;
    final private State state;
    private boolean isExpanded;
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
        if (this.children.isEmpty()) {
            System.out.println("empty");
        }
        return Collections.max(this.children, Comparator.comparing(Node::getWinScore));
    }

    public void addScore(float score) {
        this.winScore += score;
    }

    public void incrementVisitCount() {
        this.visitCount++;
    }

    public Node makeRandomMove() {
        List<State> expandedStates = this.state.getExpandedStates();
        return expandedStates.size() > 0 ? new Node(expandedStates.get(0), this) : null;
    }

    public Node makeRandomMove(Set<State> states) {
        List<State> expandedStates = this.state.getExpandedStates();
        for (State state : expandedStates) {
            if (!states.contains(state)) {
                return new Node(state, this);
            }
        }
        return null;
    }

    public void removeChild(Node child) {
        this.children.remove(child);
    }

    @Override
    public boolean equals(Object obj) {
        return obj instanceof Node && this.state.equals(((Node) obj).state);
    }

    @Override
    public int hashCode() {
        return this.state.hashCode();
    }

    @Override
    public String toString() {
        return "[visitCount: " + visitCount + ", winScore: " + winScore + ", countToRoot: " + countToRoot + "]";
    }
}
