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

    private Map<Action, Integer> numberOfTimesActionTakenMap = new HashMap<>();
    private Map<Action, Double> totalValueOfActionMap = new HashMap<>();
    private Map<Action, Double> meanValueOfActionMap = new HashMap<>();
    private Map<Action, Float> actionProbabilityMap = new HashMap<>();
    private Map<Action, Node> actionChildMap = new HashMap<>();

    private int countToRoot = 0;
    final private State state;
    private boolean isExpanded;
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
            this.numberOfTimesActionTakenMap.put(action, 0);
            this.totalValueOfActionMap.put(action, 0.0);
            this.meanValueOfActionMap.put(action, 0.0);
            this.actionProbabilityMap.put(action, (float) (1 / applicableActions[0].length));
        }
    }

    // Step 1: Choose the action that maximizes Q + U
    // Q = The mean score of the next state
    // U = explorationFactor
    public Action chooseBestAction() {
        if (this.numberOfTimesActionTakenMap.isEmpty()) throw new IllegalArgumentException("No actions possible from node");
        double bestActionScore = Integer.MIN_VALUE;
        Action bestAction = null;
        for (Action action : this.numberOfTimesActionTakenMap.keySet()) {
            double score = this.meanValueOfActionMap.get(action) + this.explorationFactor(action);
            if (score > bestActionScore) {
                bestActionScore = score;
                bestAction = action;
            }
        }
        return bestAction;
    }

    // U = c_(puct) * P(s,a) * sqrt(sum_b(N(s,b)))/(1 + N(s,a)
    private double explorationFactor(Action action) {
        return explorationProbability * this.actionProbabilityMap.get(action) *
                Math.sqrt(this.sumOfOtherAction(action)) / (1 + this.numberOfTimesActionTakenMap.get(action));
    }

    // Step 4: Select a move deterministically (with greatest N (number of times visited))
    public Node getChildWithMaxVisitCount() {
        if (this.actionChildMap.isEmpty()) throw new IllegalArgumentException("Node has no children");
        Action actionTakenMostTimes = Collections.max(this.numberOfTimesActionTakenMap.entrySet(),
                Map.Entry.comparingByValue()).getKey();
        return this.actionChildMap.get(actionTakenMostTimes);
    }

    // Step 4: Select a move stochastically
    // Probability vector of all moves, based on formula
    // prob(a|s)=(N(s,a)^(1/temp))/(sum_b(N(s,b)^(1/temp))
    public Node getChildStochastic(boolean train) {
        if (this.actionChildMap.isEmpty()) throw new IllegalArgumentException("Node has no children");
        //TODO: Add temperature
        //int temperature = train ? 1 : 0;
        Map<Action, Double> probabilityMap = new HashMap<>();
        for (Map.Entry<Action, Integer> actionEntry : this.numberOfTimesActionTakenMap.entrySet()) {
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
                return this.actionChildMap.get(actionDoubleEntry.getKey());
            }
        }
        return null;
    }

    public int sumOfOtherAction(Action action) {
        return this.numberOfTimesActionTakenMap.entrySet().stream().
                filter(e -> !e.getKey().equals(action)).
                map(Map.Entry::getValue).
                reduce(0, Integer::sum);
    }

    public Node getChildWithMaxScore() {
        if (this.actionChildMap.isEmpty()) throw new IllegalArgumentException("Node has no children");
        Action actionWithMaxScore = Collections.max(this.totalValueOfActionMap.entrySet(),
                Map.Entry.comparingByValue()).getKey();
        return this.actionChildMap.get(actionWithMaxScore);
    }

    // Step 3: W -> W + v, Q = W / N
    public void addScore(Action action, float score) {
        double newActionScore = this.totalValueOfActionMap.get(action) + score;
        this.totalValueOfActionMap.put(action, newActionScore);
        this.meanValueOfActionMap.put(action, newActionScore / this.numberOfTimesActionTakenMap.get(action));
    }

    // Step 3: N -> N + 1
    public void incrementVisitCount(Action action) {
        this.numberOfTimesActionTakenMap.put(action, this.numberOfTimesActionTakenMap.get(action) + 1);
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
        this.actionChildMap.remove(child.actionPerformed, child);
    }

    public List<Node> getChildren() {
        return new ArrayList<>(this.actionChildMap.values());
    }

    @Override
    public boolean equals(Object obj) {
        return obj instanceof Node && this.state.equals(((Node) obj).state);
    }

    @Override
    public int hashCode() {
        return this.state.hashCode();
    }

}
