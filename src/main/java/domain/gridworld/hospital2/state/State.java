package domain.gridworld.hospital2.state;

import domain.gridworld.hospital2.state.actions.IApplicableAction;
import domain.gridworld.hospital2.state.objects.stateobjects.Agent;
import domain.gridworld.hospital2.state.objects.stateobjects.Box;
import domain.gridworld.hospital2.state.objects.stateobjects.Object;
import domain.gridworld.hospital2.state.objects.*;
import domain.gridworld.hospital2.state.objects.CanvasDetails;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;
import org.javatuples.Pair;
import shared.Action;
import shared.Farge;

import java.awt.*;
import java.util.List;
import java.util.Map;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@ToString
public class State {
    private Map<Coordinate, Agent> agents;
    private Map<Coordinate, Box> boxes;

    @Getter private Set<IApplicableAction> actionsToNextState;
    @Getter private Set<Object> staticObjects;

    @Setter @Getter private long stateTime;

    public State(Map<Coordinate, Agent> agents, Map<Coordinate, Box> boxes) {
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

    public Optional<Agent> getAgentAt(Coordinate coordinate) {
        return this.agents.containsKey(coordinate) ? Optional.of(this.agents.get(coordinate)) : Optional.empty();
    }

    public Optional<Box> getBoxAt(Coordinate coordinate) {
        return this.boxes.containsKey(coordinate) ? Optional.of(this.boxes.get(coordinate)) : Optional.empty();
    }

    public void updateAgent(Agent agent, Coordinate newCoordinate) {
        this.agents.remove(agent.getCoordinate());
        agent.setCoordinate(newCoordinate);
        this.agents.put(agent.getCoordinate(), agent);
    }

    public void updateBox(Box box, Coordinate newCoordinate) {
        this.boxes.remove(box.getCoordinate());
        box.setCoordinate(newCoordinate);
        this.boxes.put(box.getCoordinate(), box);
    }

    public boolean isCellFree(Coordinate coordinate) {
        return this.getBoxAt(coordinate).isEmpty() && this.getAgentAt(coordinate).isEmpty();
    }

    private State copyOf() {
        Map<Coordinate, Agent> agents = new HashMap<>();
        this.agents.values().forEach(a -> agents.put(a.getCoordinate(), (Agent) a.clone()));
        Map<Coordinate, Box> boxes = new HashMap<>();
        this.boxes.values().forEach(b -> boxes.put(b.getCoordinate(), (Box) b.clone()));
        return new State(agents, boxes);
    }

    /**
     * Determines which actions are applicable and non-conflicting.
     * Returns an array with true for each action which was applicable and non-conflicting, and false otherwise.
     */
    public Pair<State, boolean[]> apply(Action[] jointAction, StaticState staticState) {
        boolean[] applicable = new boolean[this.agents.size()];
        IApplicableAction[] applicableActions = new IApplicableAction[this.agents.size()];
        List<Agent> agentsSorted = this.getAgents().stream().sorted(Agent::compareTo).collect(Collectors.toList());
        int agentIndex = 0;
        for (Agent agent : agentsSorted) {
            Action action = jointAction[agentIndex];
            applicableActions[agentIndex] = IApplicableAction.getInstance(action, agent, this);

            applicable[agentIndex] = applicableActions[agentIndex].isPreconditionsMet(this, staticState);

            if (!applicable[agentIndex]) continue;

            for (int prevAction = 0; prevAction < agentIndex; prevAction++) {
                if (applicableActions[agentIndex].isConflicting(applicableActions[prevAction])) {
                    applicable[agentIndex] = false;
                    applicable[prevAction] = false;
                }
            }
            agentIndex++;
        }

        State newState = this.copyOf();
        for (agentIndex = 0; agentIndex < jointAction.length; agentIndex++) {
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
        this.staticObjects.forEach(o -> {
            System.out.println("Static: " + o.getLetter());
            o.draw(g, canvasDetails);
        });
        System.out.println();
    }

    public void drawDynamicObjects(Graphics2D g, CanvasDetails canvasDetails, State nextState, double interpolation) {
        this.actionsToNextState.forEach(a -> a.draw(g, canvasDetails, nextState, interpolation));
    }
}
