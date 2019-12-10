package searchclient;

import searchclient.level.Box;
import searchclient.level.Coordinate;
import searchclient.level.DistanceMap;
import shared.Action;
import shared.Farge;

import java.util.*;
import java.util.stream.Collectors;

public class State {
    private static final Random RNG = new Random(1);

    public DistanceMap distanceMap;

    public int[] agentRows;
    public int[] agentCols;
    public Farge[] agentColors;
    private Set<Farge> agentFarges;

    public byte[][] wallsAndGoalsByteRepresentation;

    public boolean[][] walls;

    public Map<Coordinate, Character> goals;
    public Map<Coordinate, Box> boxMap;


    public final State parent;
    public final Action[] jointAction;
    private final int g;
    public int h = -1;

    private int _hash = 0;
    private Action[][] applicableActions;

    /**
     * Constructs an initial state.
     * Arguments are not copied, and therefore should not be modified after being passed in.
     */
    public State(DistanceMap distanceMap, int[] agentRows, int[] agentCols, Farge[] agentColors, boolean[][] walls,
                 Map<Coordinate, Box> boxMap, Map<Coordinate, Character> goals) {
        this.distanceMap = distanceMap;
        this.agentRows = agentRows;
        this.agentCols = agentCols;
        this.agentColors = agentColors;
        this.agentFarges = EnumSet.copyOf(Arrays.stream(agentColors).filter(Objects::nonNull).collect(Collectors.toList()));
        this.walls = walls;
        this.boxMap = boxMap;
        this.goals = goals;
        this.parent = null;
        this.jointAction = null;
        this.g = 0;
        this.createWallsAndGoalsByteRepresentation();
    }

    private void createWallsAndGoalsByteRepresentation() {
        this.wallsAndGoalsByteRepresentation = new byte[this.walls.length * 2][this.walls[0].length];
        for (Map.Entry<Coordinate, Character> entry : this.goals.entrySet()) {
            Coordinate goalCoordinate = entry.getKey();
            if ('A' <= entry.getValue() && entry.getValue() <= 'Z')
                this.wallsAndGoalsByteRepresentation[goalCoordinate.getRow()][goalCoordinate.getCol()] = 1;
            else
                this.wallsAndGoalsByteRepresentation[goalCoordinate.getRow()][goalCoordinate.getCol()] = -1;
        }
        for (int row = 0; row < this.walls.length; row++) {
            for (int col = 0; col < this.walls[row].length; col++) {
                if (!this.walls[row][col]) {
                    this.wallsAndGoalsByteRepresentation[this.walls.length + row][col] = 1;
                }
            }
        }
    }

    /**
     * Constructs the state resulting from applying jointAction in parent.
     * Precondition: Joint action must be applicable and non-conflicting in parent state.
     */
    private State(State parent, Action[] jointAction) {
        // Copy parent.
        this.distanceMap = parent.distanceMap;
        this.agentRows = Arrays.copyOf(parent.agentRows, parent.agentRows.length);
        this.agentCols = Arrays.copyOf(parent.agentCols, parent.agentCols.length);
        this.agentColors = parent.agentColors;
        this.agentFarges = parent.agentFarges;
        this.walls = parent.walls;
        this.boxMap = new HashMap<>(parent.boxMap.size());
        this.boxMap.putAll(parent.boxMap);
        this.goals = parent.goals;
        this.parent = parent;
        //parent.children.add(this);
        this.jointAction = Arrays.copyOf(jointAction, jointAction.length);
        this.g = parent.g + 1;
        this.wallsAndGoalsByteRepresentation = parent.wallsAndGoalsByteRepresentation;

        // Apply each action.
        int numAgents = this.agentRows.length;
        for (int agent = 0; agent < numAgents; ++agent) {
            Action action = jointAction[agent];

            Coordinate oldCoordinate;
            Coordinate newCoordinate;
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

                    oldCoordinate = new Coordinate(this.agentRows[agent], this.agentCols[agent]);
                    newCoordinate = new Coordinate(this.agentRows[agent] + action.getBoxDeltaRow(),
                            this.agentCols[agent] + action.getBoxDeltaCol());
                    this.moveBox(oldCoordinate, newCoordinate);
                    break;

                case Pull:
                    oldCoordinate = new Coordinate(this.agentRows[agent] + action.getBoxDeltaRow(),
                            this.agentCols[agent] + action.getBoxDeltaCol());
                    newCoordinate = new Coordinate(this.agentRows[agent], this.agentCols[agent]);
                    this.moveBox(oldCoordinate, newCoordinate);

                    this.agentRows[agent] += action.getAgentDeltaRow();
                    this.agentCols[agent] += action.getAgentDeltaCol();
                    break;
                case Paint:
                    Coordinate coordinate = new Coordinate(this.agentRows[agent] + action.getBoxDeltaRow(),
                            this.agentCols[agent] + action.getBoxDeltaCol());
                    Box oldBox = this.boxMap.remove(coordinate);
                    Box newBox = new Box(oldBox.getCharacter(), action.getColor());
                    this.boxMap.put(coordinate, newBox);
                    break;

            }
        }
    }

    public int g() {
        return this.g;
    }

    public boolean isGoalState() {
        for (Map.Entry<Coordinate, Character> goal : this.goals.entrySet()) {
            Coordinate coordinate = goal.getKey();
            Character goalCharacter = goal.getValue();
            if ('0' <= goalCharacter && goalCharacter <= '9') {
                if (this.agentRows[goalCharacter - '0'] != coordinate.getRow() ||
                        this.agentCols[goalCharacter - '0'] != coordinate.getCol()) {
                    return false;
                }
            } else {
                Box box = this.boxMap.get(coordinate);
                if (box == null || box.getCharacter() != goalCharacter) {
                    return false;
                }
            }
        }
        return true;
    }

    public Action[][] getApplicableActions() {
        if (this.applicableActions != null) return this.applicableActions;
        Action[][] applicableActions = new Action[this.agentRows.length][];
        for (int agent = 0; agent < this.agentRows.length; ++agent) {

            ArrayList<Action> agentActions = new ArrayList<>(Action.getAllActions().size());
            for (Action action : Action.getAllActions()) {
                if (this.isApplicable(agent, action)) {
                    agentActions.add(action);
                }
            }
            applicableActions[agent] = agentActions.toArray(new Action[0]);
        }
        this.applicableActions = applicableActions;
        return applicableActions;
    }

    public ArrayList<State> getExpandedStates() {
        int numAgents = this.agentRows.length;

        // Determine list of applicable actions for each individual agent.
        Action[][] applicableActions = this.getApplicableActions();

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
        Farge agentFarge = this.agentColors[agent];
        int boxRow;
        int boxCol;
        Box box;
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
                if (box == null || !agentFarge.equals(box.getColor())) {
                    return false;
                }
                destinationRow = boxRow + action.getBoxDeltaRow();
                destinationCol = boxCol + action.getBoxDeltaCol();
                return this.cellIsFree(destinationRow, destinationCol);

            case Pull:
                boxRow = agentRow + action.getBoxDeltaRow();
                boxCol = agentCol + action.getBoxDeltaCol();
                box = this.boxAt(boxRow, boxCol);
                if (box == null || !agentFarge.equals(box.getColor())) {
                    return false;
                }
                destinationRow = agentRow + action.getAgentDeltaRow();
                destinationCol = agentCol + action.getAgentDeltaCol();
                return this.cellIsFree(destinationRow, destinationCol);
            case Paint:
                if(agentFarge.equals(Farge.Grey) && agentFarges.contains(action.getColor())){
                    boxRow = agentRow + action.getBoxDeltaRow();
                    boxCol = agentCol + action.getBoxDeltaCol();
                    box = this.boxAt(boxRow, boxCol);
                    return box != null;
                }
                return false;
        }

        // Unreachable:
        return false;
    }

    private boolean moveBox(Coordinate oldCoordinate, Coordinate newCoordinate) {
        Box box = this.boxMap.remove(oldCoordinate);
        if (box != null) {
            this.boxMap.put(newCoordinate, box);
        }
        return box != null;
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
            if (jointAction[a1].getType().equals(Action.ActionType.NoOp) || jointAction[a1].getType().equals(Action.ActionType.Paint)) {
                continue;
            }

            for (int a2 = a1 + 1; a2 < numAgents; ++a2) {
                if (jointAction[a2].getType().equals(Action.ActionType.NoOp) || jointAction[a1].getType().equals(Action.ActionType.Paint)) {
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
        return !this.walls[row][col] && this.boxAt(row, col) == null && this.agentAt(row, col) == 0;
    }

    private Box boxAt(int row, int col) {
        return this.boxMap.get(new Coordinate(row, col));
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
            result = prime * result + this.boxMap.hashCode();
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
                this.boxMap.equals(other.boxMap);
    }

    @Override
    public String toString() {
        StringBuilder s = new StringBuilder();
        for (int row = 0; row < this.walls.length; row++) {
            for (int col = 0; col < this.walls[row].length; col++) {
                Box box = this.boxMap.get(new Coordinate(row, col));
                if (box != null) {
                    s.append(box.getCharacter());
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

    public String toMLString() {
        byte[][] byteRepresentation = new byte[this.wallsAndGoalsByteRepresentation.length + this.walls.length][this.walls[0].length];
        for (int i = 0; i < this.wallsAndGoalsByteRepresentation.length; i++) {
            byteRepresentation[i] = Arrays.copyOf(this.wallsAndGoalsByteRepresentation[i], this.wallsAndGoalsByteRepresentation[i].length);
        }

//        for (byte agent = 0; agent < this.agentCols.length; agent++) {
        for (int row = 0; row < this.walls.length; row++) {
            for (int col = 0; col < this.walls[row].length; col++) {
                if (this.agentRows[0] == row && this.agentCols[0] == col) {
                    byteRepresentation[this.walls.length * 2 + row][col] = 1;
                } else {
                    Coordinate coordinate = new Coordinate(row, col);
                    Box box = this.boxMap.get(coordinate);
                    if (box != null) {
                        byteRepresentation[this.walls.length * 2 + row][col] = -1;
                    }
                }

            }
//            }
        }
        return Arrays.deepToString(byteRepresentation);
    }
}
