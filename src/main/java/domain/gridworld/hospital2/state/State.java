package domain.gridworld.hospital2.state;

import domain.gridworld.hospital2.Action;
import lombok.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.function.Predicate;

@Data
@AllArgsConstructor
public class State {
    private List<Object> boxes;
    private List<Object> agents;

    private long stateTime;

    public Object getBox(int id) {
        return this.boxes.get(id);
    }

    public Object getAgent(int id) {
        return this.agents.get(id);
    }

    private Predicate<Object> getPred(int col, int row) {
        return o -> o.getCol() == col && o.getRow() == row;
    }

    public boolean cellFree(int col, int row) {
        return this.getBoxAt(col, row).isEmpty() && this.getAgentAt(col, row).isEmpty();
    }

    public Optional<Object> getBoxAt(int col, int row) {
        return this.getObjectAt(this.boxes, col, row);
    }

    public Optional<Object> getAgentAt(int col, int row) {
        return this.getObjectAt(this.agents, col, row);
    }

    private Optional<Object> getObjectAt(List<Object> list, int col, int row) {
        return list.stream().filter(getPred(col, row)).findFirst();
    }

    public State copyOf() {
        return new State(new ArrayList<>(this.boxes), new ArrayList<>(this.agents), -1);
    }

    /**
     * Determines which actions are applicable and non-conflicting.
     * Returns an array with true for each action which was applicable and non-conflicting, and false otherwise.
     */
    private boolean[] isApplicable(Action[] jointAction) {
        boolean[] applicable = new boolean[this.agents.size()];
        short[] destRows = new short[this.agents.size()];
        short[] destCols = new short[this.agents.size()];
        short[] boxRows = new short[this.agents.size()];
        short[] boxCols = new short[this.agents.size()];


        for (int agentIndex = 0; agentIndex < this.agents.size(); agentIndex++) {
            Action action = jointAction[agentIndex];
            Object agent = this.getAgent(agentIndex);
            Optional<Object> box = this.getBoxAt(
                    agent.getRow() + action.getBoxDeltaRow(), agent.getCol() + action.getBoxDeltaCol());

            destRows[agentIndex] = (short) (agent.getRow() + action.getAgentDeltaRow());
            destCols[agentIndex] = (short) (agent.getCol() + action.getAgentDeltaCol());
            if (box.isPresent()) {
                boxRows[agentIndex] = (short) (box.get().getRow() + action.getBoxDeltaRow());
                boxCols[agentIndex] = (short) (box.get().getCol() + action.getBoxDeltaCol());
            }
            switch (action.getType()) {
                case NoOp:
                    applicable[agentIndex] = true;
                    break;

                case Move:
                    applicable[agentIndex] = this.cellFree(destRows[agentIndex], destCols[agentIndex]);
                    break;

                case Push:
                case Pull:
                    applicable[agentIndex] = box.isPresent() &&
                            agent.getColor().equals(box.get().getColor()) &&
                            this.cellFree(destRows[agentIndex], destCols[agentIndex]);
                    break;
            }
        }

        // Test conflicts.
        boolean[] conflicting = new boolean[this.agents.size()];
        for (byte a1 = 0; a1 < this.agents.size(); ++a1) {
            if (!applicable[a1] || jointAction[a1].getType().equals(Action.Type.NoOp)) {
                continue;
            }
            for (byte a2 = 0; a2 < a1; ++a2) {
                if (!applicable[a2] || jointAction[a2].getType().equals(Action.Type.NoOp)) {
                    continue;
                }

                // Objects moving into same cell?
                if (destRows[a1] == destRows[a2] && destCols[a1] == destCols[a2]) {
                    conflicting[a1] = true;
                    conflicting[a2] = true;
                }

                // Moving same box?
                if (boxRows[a1] == boxRows[a2] && boxCols[a1] == boxCols[a2]) {
                    conflicting[a1] = true;
                    conflicting[a2] = true;
                }
            }
        }

        for (byte agent = 0; agent < this.agents.size(); ++agent) {
            applicable[agent] &= !conflicting[agent];
        }

        return applicable;
    }

    /**
     * Applies the actions in jointAction which are applicable to the latest state and returns the resulting state.
     * TODO: Legge til noe nytt?
     */
    private State apply(Action[] jointAction, boolean[] applicable) {
        State currentState = this.states[this.numStates - 1];
        State newState = new State(currentState);

        for (byte agent = 0; agent < jointAction.length; ++agent) {
            if (!applicable[agent]) {
                // Inapplicable or conflicting action - do nothing instead.
                continue;
            }

            Action action = jointAction[agent];
            short newAgentRow;
            short newAgentCol;
            short oldBoxRow;
            short oldBoxCol;
            short newBoxRow;
            short newBoxCol;

            switch (action.type) {
                case NoOp:
                    // Do nothing.
                    break;

                case Move:
                    newAgentRow = (short) (currentState.agentRows[agent] + action.moveDeltaRow);
                    newAgentCol = (short) (currentState.agentCols[agent] + action.moveDeltaCol);
                    this.moveAgent(newState, agent, newAgentRow, newAgentCol);
                    break;

                case Push:
                    newAgentRow = (short) (currentState.agentRows[agent] + action.boxDeltaRow);
                    newAgentCol = (short) (currentState.agentCols[agent] + action.boxDeltaCol);
                    oldBoxRow = newAgentRow;
                    oldBoxCol = newAgentCol;
                    newBoxRow = (short) (oldBoxRow + action.moveDeltaRow);
                    newBoxCol = (short) (oldBoxCol + action.moveDeltaCol);
                    this.moveBox(newState, oldBoxRow, oldBoxCol, newBoxRow, newBoxCol);
                    this.moveAgent(newState, agent, newAgentRow, newAgentCol);
                    break;

                case Pull:
                    newAgentRow = (short) (currentState.agentRows[agent] + action.moveDeltaRow);
                    newAgentCol = (short) (currentState.agentCols[agent] + action.moveDeltaCol);
                    oldBoxRow = (short) (currentState.agentRows[agent] + action.boxDeltaRow);
                    oldBoxCol = (short) (currentState.agentCols[agent] + action.boxDeltaCol);
                    newBoxRow = currentState.agentRows[agent];
                    newBoxCol = currentState.agentCols[agent];
                    this.moveAgent(newState, agent, newAgentRow, newAgentCol);
                    this.moveBox(newState, oldBoxRow, oldBoxCol, newBoxRow, newBoxCol);
                    break;

                //TODO: Implement :)
                case Paint:


            }
        }

        return newState;
    }

    /**
     * Execute a joint action.
     * Returns a boolean array with succes for each agent.
     */
    boolean[] execute(Action[] jointAction, long actionTime) {
        // Determine applicable and non-conflicting actions.
        boolean[] applicable = this.isApplicable(jointAction);

        // Create new state with applicable and non-conflicting actions.
        State newState = this.apply(jointAction, applicable);

        // Update this.states and this.numStates. Grow as necessary.
        if (this.allowDiscardingPastStates) {
            this.states[0] = newState;
            this.stateTimes[0] = actionTime;
            // NB. This change will not be visible to other threads; if we needed that we could set numStates = 1.
        } else {
            if (this.states.length == this.numStates) {
                this.states = Arrays.copyOf(this.states, this.states.length * 2);
                this.stateTimes = Arrays.copyOf(this.stateTimes, this.stateTimes.length * 2);
            }
            this.states[this.numStates] = newState;
            this.stateTimes[this.numStates] = actionTime;

            // Non-atomic increment OK, only the protocol thread may write to numStates.
            // NB. This causes visibility of the new state to other threads.
            //noinspection NonAtomicOperationOnVolatileField
            ++this.numStates;
        }

        return applicable;
    }
}
