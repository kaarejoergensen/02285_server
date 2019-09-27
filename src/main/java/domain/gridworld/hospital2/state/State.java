package domain.gridworld.hospital2.state;

import domain.gridworld.hospital2.Action;
import domain.gridworld.hospital2.state.objects.Agent;
import domain.gridworld.hospital2.state.objects.Box;
import domain.gridworld.hospital2.state.objects.Object;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;
import org.javatuples.Pair;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.function.Predicate;

@AllArgsConstructor
@ToString
public class State {
    @Setter private Map<String, Agent> agents;
    @Setter private Map<String, Box> boxes;

    @Setter @Getter private long stateTime;

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
        return new State(agents, boxes, -1);
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
            Object agent = newState.getAgent(agentIds[agentIndex]);
            agent.setRow(newAgentRows[agentIndex]);
            agent.setCol(newAgentCols[agentIndex]);

            if (boxIds[agentIndex] != null) {
                Object box = newState.getBox(boxIds[agentIndex]);
                box.setRow(newBoxRows[agentIndex]);
                box.setCol(newBoxCols[agentIndex]);
            }
        }
        return Pair.with(newState, applicable);
    }
}
