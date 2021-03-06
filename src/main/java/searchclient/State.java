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
    public static final int MAX_WIDTH = 15;
    public static final int MAX_HEIGHT = 10;

    private int heightDiff, widthDiff;

    public String levelName;
    public DistanceMap distanceMap;

    public int[] agentRows;
    public int[] agentCols;
    public Farge[] agentColors;
    private Set<Farge> agentFarges;

    public byte[][][] wallsAndGoalsByteRepresentation;

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
    public State(String levelName, DistanceMap distanceMap, int[] agentRows, int[] agentCols, Farge[] agentColors, boolean[][] walls,
                 Map<Coordinate, Box> boxMap, Map<Coordinate, Character> goals) {
        this.levelName = levelName;
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
        this.wallsAndGoalsByteRepresentation = new byte[12][MAX_HEIGHT][MAX_WIDTH];

        this.heightDiff = Math.round((MAX_HEIGHT - this.walls.length) / 2f);
        this.widthDiff = Math.round((MAX_WIDTH - this.walls[0].length) / 2f);
        if (this.heightDiff < 0 || this.widthDiff < 0) {
            System.err.println("Map too big for ML");
            return;
        }
        for (Map.Entry<Coordinate, Character> entry : this.goals.entrySet()) {
            Coordinate goalCoordinate = entry.getKey();
            if ('A' <= entry.getValue() && entry.getValue() <= 'Z') {
                this.wallsAndGoalsByteRepresentation[0][this.heightDiff + goalCoordinate.getRow()][this.widthDiff + goalCoordinate.getCol()] = 1;
            }
            else {
                this.wallsAndGoalsByteRepresentation[1][this.heightDiff + goalCoordinate.getRow()][this.widthDiff + goalCoordinate.getCol()] = 1;
            }
        }
        for (int row = 0; row < this.walls.length; row++) {
            for (int col = 0; col < this.walls[row].length; col++) {
                if (!this.walls[row][col]) {
                    this.wallsAndGoalsByteRepresentation[2][this.heightDiff + row][this.widthDiff + col] = 1;
                }
            }
        }
        for (int i = 3; i < 12; i += 3) {
            this.wallsAndGoalsByteRepresentation[i] = this.transpose(this.wallsAndGoalsByteRepresentation[0], i);
            this.wallsAndGoalsByteRepresentation[i + 1] = this.transpose(this.wallsAndGoalsByteRepresentation[1], i);
            this.wallsAndGoalsByteRepresentation[i + 2] = this.transpose(this.wallsAndGoalsByteRepresentation[2], i);
        }
    }

    /**
     * Constructs the state resulting from applying jointAction in parent.
     * Precondition: Joint action must be applicable and non-conflicting in parent state.
     */
    private State(State parent, Action[] jointAction) {
        // Copy parent.
        this.levelName = parent.levelName;
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
        this.heightDiff = parent.heightDiff;
        this.widthDiff = parent.widthDiff;

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

    public synchronized ArrayList<State> getExpandedStates() {
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
                Coordinate coordinate = new Coordinate(row, col);
                Box box = this.boxMap.get(coordinate);
                Character goal = this.goals.get(coordinate);
                if (box != null) {
                    s.append(box.getCharacter());
                } else if (goal != null) {
                    s.append(goal);
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

    public byte[][] transposeHorizontal(byte[][] in) {
        byte[][] res = new byte[in.length][in[0].length];
        for (int i = 0; i < in.length; i++) {
            res[in.length - i - 1] = Arrays.copyOf(in[i], in[i].length);
        }
        return res;
    }

    public byte[][] transposeVertical(byte[][] in) {
        byte[][] res = new byte[in.length][in[0].length];
        for (int i = 0; i < in.length; i++) {
            for (int j = 0; j < in[i].length; j++) {
                res[i][in[i].length - j - 1] = in[i][j];
            }
        }
        return res;
    }

    private byte[][] transpose(byte[][] in, int mode) {
        switch (mode) {
            case 0:
                return in;
            case 3:
                return this.transposeVertical(in);
            case 6:
                return this.transposeHorizontal(in);
            case 9:
            default:
                return this.transposeHorizontal(this.transposeVertical(in));
        }
    }


    public String toMLString() {
        if (this.heightDiff < 0 || this.widthDiff < 0) {
            throw new IllegalArgumentException("Map too big for ML");
        }
        byte[][][] byteRepresentation = new byte[5][MAX_HEIGHT][MAX_WIDTH];

        byteRepresentation[0] = this.wallsAndGoalsByteRepresentation[0];
        byteRepresentation[1] = this.wallsAndGoalsByteRepresentation[1];
        byteRepresentation[2] = this.wallsAndGoalsByteRepresentation[2];
        this.buildByteRepresentation(byteRepresentation);

        return Arrays.deepToString(byteRepresentation);
    }

    public List<String> getAllMLTranspositions() {
        if (this.heightDiff < 0 || this.widthDiff < 0) {
            throw new IllegalArgumentException("Map too big for ML");
        }
        List<String> res = new ArrayList<>();

        for (int i = 0; i < 12; i+=3) {
            byte[][][] byteRepresentation = new byte[5][MAX_HEIGHT][MAX_WIDTH];

            byteRepresentation[0] = this.wallsAndGoalsByteRepresentation[i];
            byteRepresentation[1] = this.wallsAndGoalsByteRepresentation[i + 1];
            byteRepresentation[2] = this.wallsAndGoalsByteRepresentation[i + 2];
            this.buildByteRepresentation(byteRepresentation);
            byteRepresentation[3] = this.transpose(byteRepresentation[3], i);
            byteRepresentation[4] = this.transpose(byteRepresentation[4], i);

            res.add(Arrays.deepToString(byteRepresentation));
        }

        return res;
    }

    public void buildByteRepresentation(byte[][][] in) {
        for (int row = 0; row < this.walls.length; row++) {
            for (int col = 0; col < this.walls[row].length; col++) {
                if (this.agentRows[0] == row && this.agentCols[0] == col) {
                    in[3][this.heightDiff + row][this.widthDiff + col] = 1;
                } else {
                    Coordinate coordinate = new Coordinate(row, col);
                    Box box = this.boxMap.get(coordinate);
                    if (box != null) {
                        in[4][this.heightDiff + row][this.widthDiff + col] = 1;
                    }
                }
            }
        }
    }
}
