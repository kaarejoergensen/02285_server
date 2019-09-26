package domain.gridworld.hospital2.state;

import domain.gridworld.hospital2.Action;
import domain.gridworld.hospital2.state.objects.Agent;
import domain.gridworld.hospital2.state.objects.Box;
import domain.gridworld.hospital2.state.objects.Object;
import lombok.*;
import org.javatuples.Pair;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.Predicate;

@Data
@AllArgsConstructor
@ToString
public class State {
    private List<Agent> agents;
    private List<Box> boxes;

    private long stateTime;

    public Box getBox(int id) {
        return this.boxes.get(id);
    }

    public Agent getAgent(int id) {
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

    private <T extends Object> Optional<T> getObjectAt(List<T> list, int col, int row) {
        return list.stream().filter(getPred(col, row)).findFirst();
    }

    private State copyOf() {
        List<Agent> agents = new ArrayList<>();
        this.agents.forEach(a -> agents.add((Agent) a.clone()));
        List<Box> boxes = new ArrayList<>();
        this.boxes.forEach(b -> boxes.add((Box) b.clone()));
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

        int[] agentIds = new int[this.agents.size()];
        int[] boxIds = new int[this.agents.size()];

        for (byte agentIndex = 0; agentIndex < this.agents.size(); agentIndex++) {
            Action action = jointAction[agentIndex];

            Object agent = this.getAgent(agentIndex);
            agentIds[agentIndex] = agent.getId();
            newAgentRows[agentIndex] = (short) (agent.getRow() + action.getAgentDeltaRow());
            newAgentCols[agentIndex] = (short) (agent.getCol() + action.getAgentDeltaCol());

            switch (action.getType()) {
                case NoOp: {
                    applicable[agentIndex] = true;
                    boxIds[agentIndex] = -1;
                    break;
                }
                case Move: {
                    applicable[agentIndex] = this.cellFree(newAgentRows[agentIndex], newAgentCols[agentIndex]);
                    boxIds[agentIndex] = -1;
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

                if (boxIds[agentIndex] != -1 && boxIds[prevAction] != -1
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

            if (boxIds[agentIndex] != -1) {
                Object box = newState.getBox(boxIds[agentIndex]);
                box.setRow(newBoxRows[agentIndex]);
                box.setCol(newBoxCols[agentIndex]);
            }
        }
        return Pair.with(newState, applicable);
    }
}
