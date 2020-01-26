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

    private Integer _h = null;

    private Map<Action, Integer> numberOfTimesActionTakenMap = new HashMap<>();
    private Map<Action, Double> totalValueOfActionMap = new HashMap<>();
    private Map<Action, Double> meanValueOfActionMap = new HashMap<>();
    private Map<Action, Double> actionProbabilityMap = new HashMap<>();
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
            this.actionProbabilityMap.put(action, 1.0 / applicableActions[0].length);
        }
    }

    // Step 1: Choose the action that maximizes Q + U
    // Q = The mean score of the next state
    // U = explorationFactor
    public Action chooseBestAction() {
        if (this.meanValueOfActionMap.isEmpty()) throw new IllegalArgumentException("No actions possible from node");
        double bestActionScore = Integer.MIN_VALUE;
        Action bestAction = null;
        double sumOfActionsTaken = this.sumOfActionsTaken();
        for (Action action : this.meanValueOfActionMap.keySet()) {
            if (action.getType().equals(Action.ActionType.NoOp)) continue;
            double score = this.meanValueOfActionMap.get(action) + this.explorationFactor(action, sumOfActionsTaken);
            if (score > bestActionScore) {
                bestActionScore = score;
                bestAction = action;
            }
        }
        return bestAction;
    }

    // U = c_(puct) * P(s,a) * sqrt(sum_b(N(s,b)))/(1 + N(s,a)
    private double explorationFactor(Action action, double sumOfActionsTaken) {
        return explorationProbability * this.actionProbabilityMap.get(action) *
                Math.sqrt(sumOfActionsTaken) / (1 + this.numberOfTimesActionTakenMap.get(action));
    }

    // Step 4: Select a move deterministically (with greatest N (number of times visited))
    public Node getChildWithMaxVisitCount() {
        Action actionTakenMostTimes = this.numberOfTimesActionTakenMap.entrySet().stream()
                .filter(e -> !e.getKey().getType().equals(Action.ActionType.NoOp))
                .max(Map.Entry.comparingByValue())
                .orElseThrow(() -> new IllegalArgumentException("Node has no children")).getKey();
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
        double sumOfActionsTaken = this.sumOfActionsTaken();
        for (Map.Entry<Action, Integer> actionEntry : this.numberOfTimesActionTakenMap.entrySet()) {
            Action action = actionEntry.getKey();
            double probability = (double) actionEntry.getValue() / sumOfActionsTaken;
            probabilityMap.put(action, probability);
        }
        List<Map.Entry<Action, Double>> probabilityList = new ArrayList<>(probabilityMap.entrySet());
        Collections.shuffle(probabilityList, new Random());
        double randomNumber = Math.random();
        double cumulativeProbability = 0.0;
        for (Map.Entry<Action, Double> actionDoubleEntry : probabilityList) {
            cumulativeProbability += actionDoubleEntry.getValue();
            if (randomNumber <= cumulativeProbability) {
                return this.actionChildMap.get(actionDoubleEntry.getKey());
            }
        }
        return null;
    }

    public int sumOfActionsTaken() {
        return this.numberOfTimesActionTakenMap.values().stream().
                reduce(0, Integer::sum);
    }

    public Node getChildWithMaxScore() {
        Action actionWithMaxScore = this.totalValueOfActionMap.entrySet().stream()
                .filter(e -> !e.getKey().getType().equals(Action.ActionType.NoOp))
                .max(Map.Entry.comparingByValue())
                .orElseThrow(() -> new IllegalArgumentException("Node has no children")).getKey();
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

    public void addChildren(Collection<Node> children) {
        children.forEach(c -> this.actionChildMap.put(c.actionPerformed, c));
    }

    public boolean childrenEmpty() {
        return this.actionChildMap.isEmpty();
    }

    public Collection<Node> getChildren() {
        return this.actionChildMap.values();
    }

    public Node getRandomChild() {
        Collection<Node> children = this.getChildren();
        return children.stream()
                .skip((long) (children.size() * Math.random())).findFirst()
                .orElseThrow(() -> new IllegalArgumentException("Node has no children"));
    }

    public boolean equals(final Object o) {
        if (o == this) return true;
        if (!(o instanceof Node)) return false;
        final Node other = (Node) o;
        if (!other.canEqual(this)) return false;
        final Object this$numberOfTimesActionTakenMap = this.getNumberOfTimesActionTakenMap();
        final Object other$numberOfTimesActionTakenMap = other.getNumberOfTimesActionTakenMap();
        if (!Objects.equals(this$numberOfTimesActionTakenMap, other$numberOfTimesActionTakenMap))
            return false;
        final Object this$totalValueOfActionMap = this.getTotalValueOfActionMap();
        final Object other$totalValueOfActionMap = other.getTotalValueOfActionMap();
        if (!Objects.equals(this$totalValueOfActionMap, other$totalValueOfActionMap))
            return false;
        final Object this$meanValueOfActionMap = this.getMeanValueOfActionMap();
        final Object other$meanValueOfActionMap = other.getMeanValueOfActionMap();
        if (!Objects.equals(this$meanValueOfActionMap, other$meanValueOfActionMap))
            return false;
        final Object this$actionProbabilityMap = this.getActionProbabilityMap();
        final Object other$actionProbabilityMap = other.getActionProbabilityMap();
        if (!Objects.equals(this$actionProbabilityMap, other$actionProbabilityMap))
            return false;
        if (this.getCountToRoot() != other.getCountToRoot()) return false;
        final Object this$state = this.getState();
        final Object other$state = other.getState();
        if (!Objects.equals(this$state, other$state)) return false;
        if (this.isExpanded() != other.isExpanded()) return false;
        final Object this$actionPerformed = this.getActionPerformed();
        final Object other$actionPerformed = other.getActionPerformed();
        return Objects.equals(this$actionPerformed, other$actionPerformed);
    }

    protected boolean canEqual(final Object other) {
        return other instanceof Node;
    }

    public int hashCode() {
        if (this._h == null) {
            final int PRIME = 59;
            int result = 1;
            final Object $numberOfTimesActionTakenMap = this.getNumberOfTimesActionTakenMap();
            result = result * PRIME + ($numberOfTimesActionTakenMap == null ? 43 : $numberOfTimesActionTakenMap.hashCode());
            final Object $totalValueOfActionMap = this.getTotalValueOfActionMap();
            result = result * PRIME + ($totalValueOfActionMap == null ? 43 : $totalValueOfActionMap.hashCode());
            final Object $meanValueOfActionMap = this.getMeanValueOfActionMap();
            result = result * PRIME + ($meanValueOfActionMap == null ? 43 : $meanValueOfActionMap.hashCode());
            final Object $actionProbabilityMap = this.getActionProbabilityMap();
            result = result * PRIME + ($actionProbabilityMap == null ? 43 : $actionProbabilityMap.hashCode());
            result = result * PRIME + this.getCountToRoot();
            final Object $state = this.getState();
            result = result * PRIME + ($state == null ? 43 : $state.hashCode());
            result = result * PRIME + (this.isExpanded() ? 79 : 97);
            final Object $actionPerformed = this.getActionPerformed();
            result = result * PRIME + ($actionPerformed == null ? 43 : $actionPerformed.hashCode());
            this._h = result;
        }
        return this._h;
    }
}
