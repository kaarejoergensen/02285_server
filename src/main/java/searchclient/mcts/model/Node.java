package searchclient.mcts.model;

import lombok.Getter;
import lombok.Setter;
import searchclient.State;
import shared.Action;

import java.util.*;

@Getter
@Setter
public class Node {
    public static final float explorationProbability = 1;

    private Map<Action, Integer> actionTakenMap = new HashMap<>();
    private Map<Action, Float> actionProbabilityMap = new HashMap<>();
    private int visitCount = 0;
    private float totalScore = 0;
    private float meanScore = 0;

    private int countToRoot = 0;
    final private State state;
    private boolean isExpanded;
    private List<Node> children = new ArrayList<>();
    private Node parent = null;
    private Action actionPerformed = null;

    public Node(State state, Node parent, Action actionPerformed) {
        this(state);
        this.countToRoot = parent.getCountToRoot() + 1;
        this.parent = parent;
        this.actionPerformed = actionPerformed;
    }

    public Node(State state) {
        this.state = state;
        this.initMaps();
    }

    private void initMaps() {
        Action[][] applicableActions = this.state.getApplicableActions();
        for (Action action : applicableActions[0]) {
            this.actionTakenMap.put(action, 0);
            this.actionProbabilityMap.put(action, (float) (1 / applicableActions[0].length));
        }
    }

    public Action chooseBestAction() {
        if (this.actionTakenMap.isEmpty()) throw new IllegalArgumentException("No actions possible from node");
        double bestActionScore = Integer.MIN_VALUE;
        Action bestAction = null;
        for (Action action : this.actionTakenMap.keySet()) {
            double score = this.meanScore + this.explorationFactor(action);
            if (score > bestActionScore) {
                bestActionScore = score;
                bestAction = action;
            }
        }
        return bestAction;
    }

    private double explorationFactor(Action action) {
        return explorationProbability * this.actionProbabilityMap.get(action) *
                Math.sqrt(this.sumOfOtherAction(action)) / (1 + this.actionTakenMap.get(action));
    }

    public Node getChildWithMaxVisitCount() {
        if (this.children.isEmpty()) throw new IllegalArgumentException("Node has no children");
        return Collections.max(this.children, Comparator.comparing(Node::getVisitCount));
    }

    public Node getChildStochastic(boolean train) {
        if (this.children.isEmpty()) throw new IllegalArgumentException("Node has no children");
        //TODO: Add temperature
        //int temperature = train ? 1 : 0;
        Map<Action, Double> probabilityMap = new HashMap<>();
        for (Map.Entry<Action, Integer> actionEntry : this.actionTakenMap.entrySet()) {
            Action action = actionEntry.getKey();
            double probability = (double) actionEntry.getValue() / (double) this.sumOfOtherAction(action);
            probabilityMap.put(action, probability);
        }
        List<Map.Entry<Action, Double>> probabilityList = new ArrayList<>(probabilityMap.entrySet());
        Collections.shuffle(probabilityList, new Random());
        double p = Math.random();
        double cumulativeProbability = 0.0;
        for (Map.Entry<Action, Double> actionDoubleEntry : probabilityList) {
            cumulativeProbability += actionDoubleEntry.getValue();
            if (p != cumulativeProbability) {
                Optional<Node> child = this.children.stream().filter(n -> n.actionPerformed.equals(actionDoubleEntry.getKey())).findFirst();
                if (child.isEmpty()) throw new IllegalArgumentException("Node with highest probability not simulated");
                return child.get();
            }
        }
        return null;
    }

    public int sumOfOtherAction(Action action) {
        return this.actionTakenMap.entrySet().stream().
                filter(e -> !e.getKey().equals(action)).
                map(Map.Entry::getValue).
                reduce(0, Integer::sum);
    }

    public Node getChildWithMaxScore() {
        if (this.children.isEmpty()) throw new IllegalArgumentException("Node has no children");
        return Collections.max(this.children, Comparator.comparing(Node::getTotalScore));
    }

    public void addScore(float score) {
        this.totalScore += score;
    }

    public void incrementVisitCount() {
        this.visitCount++;
    }

    public Node makeRandomMove(Set<State> states) {
        List<State> expandedStates = this.state.getExpandedStates();
        for (State state : expandedStates) {
            if (!states.contains(state)) {
                return new Node(state, this, state.jointAction[0]);
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
        return "[visitCount: " + visitCount + ", winScore: " + totalScore + ", countToRoot: " + countToRoot + "]";
    }
}
