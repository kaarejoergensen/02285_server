package domain.gridworld.hospital2.state;

import domain.gridworld.hospital2.Action;
import domain.gridworld.hospital2.state.objects.Agent;
import domain.gridworld.hospital2.state.objects.Box;
import domain.gridworld.hospital2.state.objects.Object;
import domain.gridworld.hospital2.state.objects.ui.CanvasDetails;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;
import org.javatuples.Pair;

import java.awt.*;
import java.util.*;
import java.util.function.Predicate;

@AllArgsConstructor
@ToString
public class State {
    @Setter private Map<String, Agent> agents;
    @Setter private Map<String, Box> boxes;

    @Getter @Setter private Set<String> movedAgents;
    @Getter @Setter private Set<String> movedBoxes;

    @Setter @Getter private long stateTime;

    public Collection<Agent> getAgents() {
        return this.agents.values();
    }

    public Collection<Box> getBoxes() {
        return this.boxes.values();
    }

    private Box getBox(String id) {
        return this.boxes.get(id);
    }

    private Agent getAgent(String id) {
        return this.agents.get(id);
    }

    private void moveBox(String id, short newRow, short newCol) {
        this.moveObject(id, newRow, newCol, this.boxes, this.movedBoxes);
    }

    private void moveAgent(String id, short newRow, short newCol) {
        this.moveObject(id, newRow, newCol, this.agents, this.movedAgents);
    }

    private void moveObject(String id, short newRow, short newCol, Map<String, ? extends Object> objects, Set<String> moved) {
        Object object = objects.get(id);
        object.setRow(newRow);
        object.setCol(newCol);
        moved.add(id);
    }

    private Predicate<Object> getPred(int col, int row) {
        return o -> o.getCol() == col && o.getRow() == row;
    }

    private boolean cellFree(int col, int row) {
        return this.getBoxAt(col, row).isEmpty() && this.getAgentAt(col, row).isEmpty();
    }

    public Optional<Box> getBoxAt(int col, int row) {
        return this.getObjectAt(this.boxes, col, row);
    }

    public Optional<Agent> getAgentAt(int col, int row) {
        return this.getObjectAt(this.agents, col, row);
    }

    private <T extends Object> Optional<T> getObjectAt(Map<String, T> map, int col, int row) {
        return map.values().stream().filter(getPred(col, row)).findFirst();
    }

    private State copyOf() {
        Map<String, Agent> agents = new HashMap<>();
        this.agents.values().forEach(a -> agents.put(a.getId(), (Agent) a.clone()));
        Map<String, Box> boxes = new HashMap<>();
        this.boxes.values().forEach(b -> boxes.put(b.getId(), (Box) b.clone()));
        return new State(agents, boxes, new HashSet<>(agents.values().size()), new HashSet<>(boxes.values().size()), -1);
    }

    /**
     * Determines which actions are applicable and non-conflicting.
     * Returns an array with true for each action which was applicable and non-conflicting, and false otherwise.
     */
    public Pair<State, boolean[]> apply(Action[] jointAction) {
        boolean[] applicable = new boolean[this.agents.size()];

        short[] newAgentRows = new short[this.agents.size()];
        short[] newAgentCols = new short[this.agents.size()];
        short[] newBoxRows = new short[this.agents.size()];
        short[] newBoxCols = new short[this.agents.size()];

        String[] agentIds = new String[this.agents.size()];
        String[] boxIds = new String[this.agents.size()];

        for (byte agentIndex = 0; agentIndex < this.agents.size(); agentIndex++) {
            Action action = jointAction[agentIndex];

            Object agent = this.getAgent("A" + agentIndex);
            agentIds[agentIndex] = agent.getId();
            newAgentRows[agentIndex] = (short) (agent.getRow() + action.getAgentDeltaRow());
            newAgentCols[agentIndex] = (short) (agent.getCol() + action.getAgentDeltaCol());

            switch (action.getType()) {
                case NoOp: {
                    applicable[agentIndex] = true;
                    boxIds[agentIndex] = null;
                    agentIds[agentIndex] = null;
                    break;
                }
                case Move: {
                    applicable[agentIndex] = this.cellFree(newAgentRows[agentIndex], newAgentCols[agentIndex]);
                    boxIds[agentIndex] = null;
                    break;
                }
                case Push: {
                    Optional<Box> box = this.getBoxAt(newAgentCols[agentIndex], newAgentRows[agentIndex]);
                    if (box.isPresent()) {
                        newBoxRows[agentIndex] = (short) (box.get().getRow() + action.getBoxDeltaRow());
                        newBoxCols[agentIndex] = (short) (box.get().getCol() + action.getBoxDeltaCol());
                        boxIds[agentIndex] = box.get().getId();
                    }
                    applicable[agentIndex] = box.isPresent() &&
                            agent.getColor().equals(box.get().getColor());
                    break;
                }
                case Pull:
                    Optional<Box> box = this.getBoxAt(
                            agent.getCol() + Math.negateExact(action.getBoxDeltaCol()),
                            agent.getRow() + Math.negateExact(action.getBoxDeltaRow()));
                    if (box.isPresent()) {
                        newBoxRows[agentIndex] = agent.getRow();
                        newBoxCols[agentIndex] = agent.getCol();
                        boxIds[agentIndex] = box.get().getId();
                    }
                    applicable[agentIndex] = box.isPresent() &&
                            agent.getColor().equals(box.get().getColor());
                    break;
            }
            if (!applicable[agentIndex] || jointAction[agentIndex].getType().equals(Action.Type.NoOp)) {
                continue;
            }
            for (int prevAction = 0; prevAction < agentIndex; prevAction++) {
                if (!applicable[prevAction] || jointAction[prevAction].getType().equals(Action.Type.NoOp)) {
                    continue;
                }

                // Objects moving into same cell?
                if (newAgentRows[agentIndex] == newAgentRows[prevAction]
                    && newAgentCols[agentIndex] == newAgentCols[prevAction]) {
                    applicable[agentIndex] = false;
                    applicable[prevAction] = false;
                }

                if (boxIds[agentIndex] != null && boxIds[prevAction] != null
                    && newBoxRows[agentIndex] == newBoxRows[prevAction]
                    && newBoxCols[agentIndex] == newBoxCols[prevAction]) {
                    applicable[agentIndex] = false;
                    applicable[prevAction] = false;
                }
            }
        }

        State newState = this.copyOf();

        for (int agentIndex = 0; agentIndex < jointAction.length; agentIndex++) {
            if (!applicable[agentIndex]) continue;
            if (agentIds[agentIndex] != null) {
                newState.moveAgent(agentIds[agentIndex], newAgentRows[agentIndex], newAgentCols[agentIndex]);
            }

            if (boxIds[agentIndex] != null) {
                newState.moveBox(boxIds[agentIndex], newBoxRows[agentIndex], newBoxCols[agentIndex]);
            }
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

    public void drawStaticObjects(Graphics2D g, CanvasDetails canvasDetails, State nextState) {
        nextState.getAgents().stream()
                .map(Object::getId)
                .filter(a -> !nextState.getMovedAgents().contains(a))
                .forEach(a -> this.getAgent(a).draw(g, canvasDetails));

        nextState.getBoxes().stream()
                .map(Object::getId)
                .filter(b -> !nextState.getMovedBoxes().contains(b))
                .forEach(b -> this.getBox(b).draw(g, canvasDetails));
    }

    public void drawDynamicObjects(Graphics2D g, CanvasDetails canvasDetails, State nextState, double interpolation) {
        for (String agentId : nextState.getMovedAgents()) {
            Agent oldAgent = this.getAgent(agentId);
            Agent newAgent = nextState.getAgent(agentId);
            if (interpolation != 0.0) this.drawAgentArm(g, canvasDetails, nextState, interpolation, newAgent, oldAgent);
            oldAgent.draw(g, canvasDetails, newAgent.getRow(), newAgent.getCol(), interpolation);
        }
        for (String boxId : nextState.getMovedBoxes()) {
            Box oldBox = this.getBox(boxId);
            Box newBox = nextState.getBox(boxId);
            oldBox.draw(g, canvasDetails, newBox.getRow(), newBox.getCol(), interpolation);
        }
    }

    private void drawAgentArm(Graphics2D g, CanvasDetails canvasDetails, State nextState, double interpolation,
                              Agent newAgent, Agent oldAgent) {
        for (String boxId : nextState.getMovedBoxes()) {
            Box oldBox = this.getBox(boxId);
            Box newBox = nextState.getBox(boxId);
            if (newAgent.getRow() == oldBox.getRow() && newAgent.getCol() == oldBox.getCol() ||
                    oldAgent.getRow() == newBox.getRow() && oldAgent.getCol() == newBox.getCol()) {
                oldAgent.drawArmPullPush(g, canvasDetails, newAgent, oldBox, newBox, interpolation);
                return;
            }
        }
        oldAgent.drawArmMove(g, canvasDetails, newAgent, interpolation);
    }
}
