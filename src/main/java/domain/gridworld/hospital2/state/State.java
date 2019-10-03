package domain.gridworld.hospital2.state;

import domain.gridworld.hospital2.state.actions.IApplicableAction;
import domain.gridworld.hospital2.state.objects.Object;
import domain.gridworld.hospital2.state.objects.*;
import domain.gridworld.hospital2.state.objects.ui.CanvasDetails;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;
import org.javatuples.Pair;
import shared.Action;

import java.awt.*;
import java.util.Map;
import java.util.*;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@ToString
public class State {
    private Map<String, Agent> agents;
    private Map<String, Box> boxes;

    @Getter private Set<IApplicableAction> actionsToNextState;
    @Getter private Set<Object> staticObjects;

    @Setter @Getter private long stateTime;

    public State(Map<String, Agent> agents, Map<String, Box> boxes) {
        this.agents = agents;
        this.boxes = boxes;
        this.actionsToNextState = new HashSet<>();
        this.staticObjects = Stream.concat(agents.values().stream(), boxes.values().stream()).collect(Collectors.toSet());
    }

    public Collection<Agent> getAgents() {
        return this.agents.values();
    }

    public Collection<Box> getBoxes() {
        return this.boxes.values();
    }

    public Box getBox(String id) {
        return this.boxes.get(id);
    }

    public Agent getAgent(String id) {
        return this.agents.get(id);
    }

    private Predicate<Object> getPred(Coordinate coordinate) {
        return o -> o.getCoordinate().equals(coordinate);
    }

    public boolean isCellFree(Coordinate coordinate) {
        return this.getBoxAt(coordinate).isEmpty() && this.getAgentAt(coordinate).isEmpty();
    }

    public Optional<Box> getBoxAt(Coordinate coordinate) {
        return this.getObjectAt(this.boxes, coordinate);
    }

    public Optional<Agent> getAgentAt(Coordinate coordinate) {
        return this.getObjectAt(this.agents, coordinate);
    }

    private <T extends Object> Optional<T> getObjectAt(Map<String, T> map, Coordinate coordinate) {
        return map.values().stream().filter(getPred(coordinate)).findFirst();
    }

    private State copyOf() {
        Map<String, Agent> agents = new HashMap<>();
        this.agents.values().forEach(a -> agents.put(a.getId(), (Agent) a.clone()));
        Map<String, Box> boxes = new HashMap<>();
        this.boxes.values().forEach(b -> boxes.put(b.getId(), (Box) b.clone()));
        return new State(agents, boxes);
    }

    /**
     * Determines which actions are applicable and non-conflicting.
     * Returns an array with true for each action which was applicable and non-conflicting, and false otherwise.
     */
    public Pair<State, boolean[]> apply(Action[] jointAction, StaticState staticState) {
        boolean[] applicable = new boolean[this.agents.size()];
        IApplicableAction[] applicableActions = new IApplicableAction[this.agents.size()];

        for (byte agentIndex = 0; agentIndex < this.agents.size(); agentIndex++) {
            Action action = jointAction[agentIndex];
            Agent agent = this.getAgent("A" + agentIndex);
            applicableActions[agentIndex] = IApplicableAction.getInstance(action, agent, this);

            applicable[agentIndex] = applicableActions[agentIndex].isPreconditionsMet(this, staticState);

            if (!applicable[agentIndex]) continue;

            for (int prevAction = 0; prevAction < agentIndex; prevAction++) {
                if (applicableActions[agentIndex].isConflicting(applicableActions[prevAction])) {
                    applicable[agentIndex] = false;
                    applicable[prevAction] = false;
                }
            }
        }

        State newState = this.copyOf();
        for (int agentIndex = 0; agentIndex < jointAction.length; agentIndex++) {
            if (!applicable[agentIndex]) continue;
            applicableActions[agentIndex].apply(newState);
            applicableActions[agentIndex].getAffectedObjects().forEach(o -> this.staticObjects.remove(o));
            this.actionsToNextState.add(applicableActions[agentIndex]);
        }
        return Pair.with(newState, applicable);
    }

    public void initObjects(CanvasDetails canvasDetails) {
        this.getBoxes().forEach(b -> b.letterTextUpdate(canvasDetails));
        this.getAgents().forEach(a -> {
            a.letterTextUpdate(canvasDetails);
            a.setArmsSize(canvasDetails);
        });
    }

    public void drawAllObjects(Graphics2D g, CanvasDetails canvasDetails, State nextState, double interpolation) {
        this.drawDynamicObjects(g, canvasDetails, nextState, interpolation);
        this.drawStaticObjects(g, canvasDetails);
    }

    public void drawStaticObjects(Graphics2D g, CanvasDetails canvasDetails) {
        this.staticObjects.forEach(o -> o.draw(g, canvasDetails));
    }

    public void drawDynamicObjects(Graphics2D g, CanvasDetails canvasDetails, State nextState, double interpolation) {
        this.actionsToNextState.forEach(a -> a.draw(g, canvasDetails, this, nextState, interpolation));
    }
}
