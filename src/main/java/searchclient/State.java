package searchclient;

import shared.Action;
import shared.Farge;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

public class State {
    private static final Random RNG = new Random(1);

    // Contains the (row, col) pair and color for each agent indexed by agent number.
    public int[] agentRows;
    public int[] agentCols;
    public Farge[] agentColors;

    // Arrays are indexed from the top-left of the level, with first index being row and second being column.
    // Row 0: (0,0) (0,1) (0,2) (0,3) ...
    // Row 1: (1,0) (1,1) (1,2) (1,3) ...
    // Row 2: (2,0) (2,1) (2,2) (2,3) ...
    // ...
    //
    // E.g. this.walls[2] is an array of booleans for the third row.
    // this.walls[row][col] is true if there's a wall at (row, col).
    public boolean[][] walls;
    public char[][] boxes;
    public Farge[] allColors;
    public int[] boxColors;
    public char[][] goals;


    public final State parent;
    public final Action[] jointAction;
    private final int g;

    private int _hash = 0;

    /**
     * Constructs an initial state.
     * Arguments are not copied, and therefore should not be modified after being passed in.
     */
    public State(int[] agentRows, int[] agentCols, Farge[] agentColors, boolean[][] walls,
                 char[][] boxes, Farge[] allColors, int[] boxColors, char[][] goals) {
        this.agentRows = agentRows;
        this.agentCols = agentCols;
        this.agentColors = agentColors;
        this.walls = walls;
        this.boxes = boxes;
        this.allColors = allColors;
        this.boxColors = boxColors;
        this.goals = goals;
        this.parent = null;
        this.jointAction = null;
        this.g = 0;
    }

    /**
     * Constructs the state resulting from applying jointAction in parent.
     * Precondition: Joint action must be applicable and non-conflicting in parent state.
     */
    private State(State parent, Action[] jointAction) {
        // Copy parent.
        this.agentRows = Arrays.copyOf(parent.agentRows, parent.agentRows.length);
        this.agentCols = Arrays.copyOf(parent.agentCols, parent.agentCols.length);
        this.agentColors = parent.agentColors;
        this.walls = parent.walls;
        this.boxes = new char[parent.boxes.length][];
        for (int i = 0; i < parent.boxes.length; i++) {
            this.boxes[i] = Arrays.copyOf(parent.boxes[i], parent.boxes[i].length);
        }
        this.allColors = parent.allColors;
        this.boxColors = Arrays.copyOf(parent.boxColors, parent.boxColors.length);
        this.goals = parent.goals;
        this.parent = parent;
        this.jointAction = Arrays.copyOf(jointAction, jointAction.length);
        this.g = parent.g + 1;

        // Apply each action.
        int numAgents = this.agentRows.length;
        for (int agent = 0; agent < numAgents; ++agent) {
            Action action = jointAction[agent];
            char box;

            switch (action.getType()) {
                case NoOp:
                    break;

                case Move:
                    this.agentRows[agent] += action.getAgentDeltaRow();
                    this.agentCols[agent] += action.getAgentDeltaCol();
                    break;

                case Push:
                    this.agentRows[agent] += action.getAgentDeltaRow();
                    this.agentCols[agent] += action.getAgentDeltaCol();
                    box = this.boxAt(this.agentRows[agent], this.agentCols[agent]);
                    this.boxes[this.agentRows[agent]][this.agentCols[agent]] = 0;
                    this.boxes[this.agentRows[agent] + action.getBoxDeltaRow()]
                            [this.agentCols[agent] + action.getBoxDeltaCol()] = box;
                    break;

                case Pull:
                    box = this.boxAt(this.agentRows[agent] + action.getBoxDeltaRow(),
                            this.agentCols[agent] + action.getBoxDeltaCol());
                    this.boxes[this.agentRows[agent] + action.getBoxDeltaRow()]
                            [this.agentCols[agent] + action.getBoxDeltaCol()] = 0;
                    this.boxes[this.agentRows[agent]][this.agentCols[agent]] = box;
                    this.agentRows[agent] += action.getAgentDeltaRow();
                    this.agentCols[agent] += action.getAgentDeltaCol();
                    break;
                case Paint:
                    char index = this.boxAt(this.agentRows[agent] + action.getBoxDeltaRow(),
                            this.agentCols[agent] + action.getBoxDeltaCol());
                    int newIndex = (this.boxColors[index - 'A'] + 1) % this.allColors.length;
                    if (this.allColors[newIndex].equals(Farge.Grey)) {
                        newIndex = (newIndex + 1) % allColors.length;
                    }
                    boxColors[index - 'A'] = newIndex;
                    break;

            }
        }
    }

    public int g() {
        return this.g;
    }

    public boolean isGoalState() {
        for (int row = 1; row < this.goals.length - 1; row++) {
            for (int col = 1; col < this.goals[row].length - 1; col++) {
                char goal = this.goals[row][col];

                if ('A' <= goal && goal <= 'Z' && this.boxes[row][col] != goal) {
                    return false;
                } else if ('0' <= goal && goal <= '9' &&
                        !(this.agentRows[goal - '0'] == row && this.agentCols[goal - '0'] == col)) {
                    return false;
                }
            }
        }
        return true;
    }

    public ArrayList<State> getExpandedStates() {
        int numAgents = this.agentRows.length;

        // Determine list of applicable actions for each individual agent.
        Action[][] applicableActions = new Action[numAgents][];
        for (int agent = 0; agent < numAgents; ++agent) {

            ArrayList<Action> agentActions = new ArrayList<>(Action.values().length);
            for (Action action : Action.values()) {
                if (this.isApplicable(agent, action)) {
                    agentActions.add(action);
                }
            }
            applicableActions[agent] = agentActions.toArray(new Action[0]);
        }

        // Iterate over joint actions, check conflict and generate child states.
        Action[] jointAction = new Action[numAgents];
        int[] actionsPermutation = new int[numAgents];
        ArrayList<State> expandedStates = new ArrayList<>(16);
        while (true) {
            for (int agent = 0; agent < numAgents; ++agent) {
                jointAction[agent] = applicableActions[agent][actionsPermutation[agent]];
            }

            if (!this.isConflicting(jointAction)) {
                expandedStates.add(new State(this, jointAction));
            }

            // Advance permutation.
            boolean done = false;
            for (int agent = 0; agent < numAgents; ++agent) {
                if (actionsPermutation[agent] < applicableActions[agent].length - 1) {
                    ++actionsPermutation[agent];
                    break;
                } else {
                    actionsPermutation[agent] = 0;
                    if (agent == numAgents - 1) {
                        done = true;
                    }
                }
            }

            // Last permutation?
            if (done) {
                break;
            }
        }

        Collections.shuffle(expandedStates, State.RNG);
        return expandedStates;
    }

    private boolean isApplicable(int agent, Action action) {
        int agentRow = this.agentRows[agent];
        int agentCol = this.agentCols[agent];
        Farge agentColors = this.agentColors[agent];
        int boxRow;
        int boxCol;
        char box;
        int destinationRow;
        int destinationCol;
        switch (action.getType()) {
            case NoOp:
                return true;

            case Move:
                destinationRow = agentRow + action.getAgentDeltaRow();
                destinationCol = agentCol + action.getAgentDeltaCol();
                return this.cellIsFree(destinationRow, destinationCol);

            case Push:
                boxRow = agentRow + action.getAgentDeltaRow();
                boxCol = agentCol + action.getAgentDeltaCol();
                box = this.boxAt(boxRow, boxCol);
                if (box == 0 || !agentColors.equals(allColors[boxColors[box - 'A']])) {
                    return false;
                }
                destinationRow = boxRow + action.getBoxDeltaRow();
                destinationCol = boxCol + action.getBoxDeltaCol();
                return this.cellIsFree(destinationRow, destinationCol);

            case Pull:
                boxRow = agentRow + action.getBoxDeltaRow();
                boxCol = agentCol + action.getBoxDeltaCol();
                box = this.boxAt(boxRow, boxCol);
                if (box == 0 || !agentColors.equals(allColors[boxColors[box - 'A']])) {
                    return false;
                }
                destinationRow = agentRow + action.getAgentDeltaRow();
                destinationCol = agentCol + action.getAgentDeltaCol();
                return this.cellIsFree(destinationRow, destinationCol);
            case Paint:
                if(agentColors.equals(Farge.Grey)){
                    boxRow = agentRow + action.getBoxDeltaRow();
                    boxCol = agentCol + action.getBoxDeltaCol();
                    box = this.boxAt(boxRow, boxCol);
                    return box != 0;
                }
                return false;
        }

        // Unreachable:
        return false;
    }

    private boolean isConflicting(Action[] jointAction) {
        int numAgents = this.agentRows.length;

        int[] destinationRows = new int[numAgents];
        int[] destinationCols = new int[numAgents];
        int[] boxRows = new int[numAgents];
        int[] boxCols = new int[numAgents];

        // Collect cells to be occupied and boxes to be moved.
        for (int agent = 0; agent < numAgents; ++agent) {
            Action action = jointAction[agent];
            int agentRow = this.agentRows[agent];
            int agentCol = this.agentCols[agent];
            int boxRow;
            int boxCol;

            switch (action.getType()) {
                case Move:
                    destinationRows[agent] = agentRow + action.getAgentDeltaRow();
                    destinationCols[agent] = agentCol + action.getAgentDeltaCol();
                    boxRows[agent] = agentRow; // Distinct dummy value.
                    boxCols[agent] = agentCol; // Distinct dummy value.
                    break;

                case Push:
                    boxRow = agentRow + action.getAgentDeltaRow();
                    boxCol = agentCol + action.getAgentDeltaCol();
                    boxRows[agent] = boxRow;
                    boxCols[agent] = boxCol;
                    destinationRows[agent] = boxRow + action.getBoxDeltaRow();
                    destinationCols[agent] = boxCol + action.getBoxDeltaCol();
                    break;

                case Pull:
                    boxRow = agentRow + action.getBoxDeltaRow();
                    boxCol = agentCol + action.getBoxDeltaCol();
                    boxRows[agent] = boxRow;
                    boxCols[agent] = boxCol;
                    destinationRows[agent] = agentRow + action.getAgentDeltaRow();
                    destinationCols[agent] = agentCol + action.getAgentDeltaCol();
                    break;
                case Paint:
                    boxRow = agentRow + action.getBoxDeltaRow();
                    boxCol = agentCol + action.getBoxDeltaCol();
                    boxRows[agent] = boxRow;
                    boxCols[agent] = boxCol;
                default:
                    break;
            }
        }

        for (int a1 = 0; a1 < numAgents; ++a1) {
            if (jointAction[a1] == Action.NoOp || jointAction[a1].getType().equals(Action.ActionType.Paint)) {
                continue;
            }

            for (int a2 = a1 + 1; a2 < numAgents; ++a2) {
                if (jointAction[a2] == Action.NoOp || jointAction[a1].getType().equals(Action.ActionType.Paint)) {
                    continue;
                }

                // Moving into same cell?
                if (destinationRows[a1] == destinationRows[a2] && destinationCols[a1] == destinationCols[a2]) {
                    return true;
                }

                // Moving same box?
                if (boxRows[a1] == boxRows[a2] && boxCols[a1] == boxCols[a2]) {
                    return true;
                }
            }
        }

        return false;
    }

    private boolean cellIsFree(int row, int col) {
        return !this.walls[row][col] && this.boxAt(row, col) == 0 && this.agentAt(row, col) == 0;
    }

    private char boxAt(int row, int col) {
        return this.boxes[row][col];
    }

    private char agentAt(int row, int col) {
        for (int i = 0; i < this.agentRows.length; i++) {
            if (this.agentRows[i] == row && this.agentCols[i] == col) {
                return (char) ('0' + i);
            }
        }
        return 0;
    }

    public Action[][] extractPlan() {
        Action[][] plan = new Action[this.g][];
        State s = this;
        while (s.jointAction != null) {
            plan[s.g - 1] = s.jointAction;
            s = s.parent;
        }
        return plan;
    }

    @Override
    public int hashCode() {
        if (this._hash == 0) {
            final int prime = 31;
            int result = 1;
            result = prime * result + Arrays.hashCode(this.agentRows);
            result = prime * result + Arrays.hashCode(this.agentCols);
            result = prime * result + Arrays.hashCode(this.boxColors);
            result = prime * result + Arrays.deepHashCode(this.boxes);
            this._hash = result;
        }
        return this._hash;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null) {
            return false;
        }
        if (this.getClass() != obj.getClass()) {
            return false;
        }
        State other = (State) obj;
        return Arrays.equals(this.agentRows, other.agentRows) &&
                Arrays.equals(this.agentCols, other.agentCols) &&
                Arrays.deepEquals(this.boxes, other.boxes) &&
                Arrays.equals(this.boxColors, other.boxColors);
    }

    @Override
    public String toString() {
        StringBuilder s = new StringBuilder();
        for (int row = 0; row < this.walls.length; row++) {
            for (int col = 0; col < this.walls[row].length; col++) {
                if (this.boxes[row][col] > 0) {
                    s.append(this.boxes[row][col]);
                } else if (this.walls[row][col]) {
                    s.append("+");
                } else if (this.agentAt(row, col) != 0) {
                    s.append(this.agentAt(row, col));
                } else {
                    s.append(" ");
                }
            }
            s.append("\n");
        }
        return s.toString();
    }

}
