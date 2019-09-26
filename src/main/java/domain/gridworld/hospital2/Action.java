package domain.gridworld.hospital2;

import lombok.Getter;

public class Action {
    public enum Type {
        NoOp,
        Move,
        Push,
        Pull,
        Paint
    }

    enum MoveDirection {
        NORTH(-1, 0),
        EAST(0, 1),
        SOUTH(1, 0),
        WEST(0, -1),
        NONE(0,0);

        @Getter private final short deltaRow;
        @Getter private final short deltaCol;

        MoveDirection(int deltaRow, int deltaCol) {
            this.deltaRow = (short) deltaRow;
            this.deltaCol = (short) deltaCol;
        }

        public static MoveDirection fromShortString(String s) {
            switch (s) {
                case "N": return NORTH;
                case "E": return EAST;
                case "S": return SOUTH;
                case "W": return WEST;
                default: return NONE;
            }
        }
    }

    @Getter private final Type type;
    private final MoveDirection agentMoveDirection;
    private final MoveDirection boxMoveDirection;

    public Action(Type type, MoveDirection agentMoveDirection, MoveDirection boxMoveDirection) {
        this.type = type;
        this.agentMoveDirection = agentMoveDirection;
        this.boxMoveDirection = boxMoveDirection;
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
        try {
            int endIndex = action.indexOf('(');
            Type type = Type.valueOf(action.substring(0, endIndex != -1 ? endIndex : action.length()));
            String[] moveDirections = action.replaceAll(".*\\(|\\).*", "").split(",");
            MoveDirection agentMoveDirection = MoveDirection.fromShortString(moveDirections[0]);
            MoveDirection boxMoveDirection;
            if (moveDirections.length > 1) {
                boxMoveDirection = MoveDirection.fromShortString(moveDirections[1]);
            } else {
                boxMoveDirection = MoveDirection.NONE;
            }
            return new Action(type, agentMoveDirection, boxMoveDirection);
        } catch (IllegalArgumentException e) {
            return null;
        }
    }
}
