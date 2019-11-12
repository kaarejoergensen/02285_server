package shared;

import lombok.Getter;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;


@Getter
public class Action {
    @Getter
    private static final List<Action> allActions = generateALlActions();

    private String name;
    private final ActionType type;
    private MoveDirection agentMoveDirection = MoveDirection.NONE;
    private MoveDirection boxMoveDirection = MoveDirection.NONE;

    private Farge color;

    private Action(ActionType type, MoveDirection agentMoveDirection, MoveDirection boxMoveDirection) {
        this.type = type;
        this.agentMoveDirection = agentMoveDirection;
        this.boxMoveDirection = boxMoveDirection;
        this.name = this.generateName();
    }

    private Action(ActionType type) {
        this.type = type;
        this.name = this.generateName();
    }

    private Action(ActionType type, MoveDirection agentMoveDirection) {
        this.type = type;
        this.agentMoveDirection = agentMoveDirection;
        this.name = this.generateName();
    }


    private Action(ActionType type, MoveDirection boxMoveDirection, Farge color) {
        this.type = type;
        this.boxMoveDirection = boxMoveDirection;
        this.color = color;
        this.name = this.generateName();
    }

    private String generateName(){
        if (type.equals(ActionType.NoOp)) return type.name();
        StringBuilder stringBuilder = new StringBuilder(type.name());
        stringBuilder.append("(").append(agentMoveDirection.getLetter());
        if(boxMoveDirection != MoveDirection.NONE){
            stringBuilder.append(",").append(boxMoveDirection.getLetter());
        }
        if (color != null) {
            stringBuilder.append(",").append(color.name());
        }
        stringBuilder.append(")");
        return stringBuilder.toString();
    }

    public short getAgentDeltaRow() {
        return agentMoveDirection.getDeltaRow();
    }

    public short getAgentDeltaCol() {
        return agentMoveDirection.getDeltaCol();

    }

    public short getBoxDeltaRow() {
        return boxMoveDirection.getDeltaRow();
    }

    public short getBoxDeltaCol() {
        return boxMoveDirection.getDeltaCol();
    }

    public static Action parse(String action) {
        for(Action a: allActions){
            if(a.getName().equals(action)) return a;
        }
        return generateNoOp();
    }

    public static List<Action> generateALlActions() {
        List<Action> actions = new ArrayList<>();
        actions.add(generateNoOp());
        actions.addAll(generateMove());
        actions.addAll(generatePush());
        actions.addAll(generatePull());
        actions.addAll(generatePaint());
        return actions;
    }

    public static Action generateNoOp() {
        return new Action(ActionType.NoOp);
    }

    public static List<Action> generateMove() {
        List<Action> actions = new ArrayList<>();
        for (MoveDirection moveDirection : MoveDirection.values()) {
            actions.add(new Action(ActionType.Move, moveDirection));
        }
        return actions;
    }

    public static List<Action> generatePush() {
        return generatePullOrPush(ActionType.Push);
    }

    public static List<Action> generatePull() {
        return generatePullOrPush(ActionType.Pull);
    }

    private static List<Action> generatePullOrPush(ActionType type) {
        List<Action> actions = new ArrayList<>();
        for (MoveDirection moveDirection : MoveDirection.values()) {
            for (MoveDirection moveDirection1 : MoveDirection.values()) {
                if (type.equals(ActionType.Pull) && moveDirection.equals(moveDirection1)) continue;
                actions.add(new Action(type, moveDirection, moveDirection1));
            }
        }
        return actions;
    }

    public static List<Action> generatePaint() {
        List<Action> actions = new ArrayList<>();
        for (MoveDirection moveDirection : MoveDirection.values()) {
            for (Object farge : Farge.getClientFarger()) {
                actions.add(new Action(ActionType.Paint, moveDirection, (Farge) farge));
            }
        }
        return actions;
    }

    public enum ActionType {
        NoOp,
        Move,
        Push,
        Pull,
        Paint
    }

    @Getter
    public enum MoveDirection {
        NORTH("N",-1, 0),
        EAST("E",0, 1),
        SOUTH("S",1, 0),
        WEST("W", 0, -1),
        NONE("None", 0,0);

        private String letter;
        private final short deltaRow;
        private final short deltaCol;

        MoveDirection(String letter, int deltaRow, int deltaCol) {
            this.letter = letter;
            this.deltaRow = (short) deltaRow;
            this.deltaCol = (short) deltaCol;
        }


    }
}