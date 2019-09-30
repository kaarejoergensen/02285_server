package shared;

import lombok.Getter;


@Getter
public enum Action {

    NoOp(ActionType.NoOp, MoveDirection.NONE,  MoveDirection.NONE),

    MoveN(ActionType.Move, MoveDirection.NORTH, MoveDirection.NONE),
    MoveS(ActionType.Move, MoveDirection.SOUTH, MoveDirection.NONE),
    MoveE(ActionType.Move, MoveDirection.EAST, MoveDirection.NONE),
    MoveW(ActionType.Move, MoveDirection.WEST, MoveDirection.NONE),

    //PUSH
    PushNN(ActionType.Push, MoveDirection.NORTH, MoveDirection.NORTH),
    PushNE(ActionType.Push, MoveDirection.NORTH, MoveDirection.EAST),
    PushNW(ActionType.Push, MoveDirection.NORTH, MoveDirection.WEST),

    PushSS(ActionType.Push, MoveDirection.SOUTH, MoveDirection.SOUTH),
    PushSE(ActionType.Push, MoveDirection.SOUTH, MoveDirection.EAST),
    PushSW(ActionType.Push, MoveDirection.SOUTH, MoveDirection.WEST),

    PushEE(ActionType.Push, MoveDirection.EAST, MoveDirection.EAST),
    PushEN(ActionType.Push, MoveDirection.EAST, MoveDirection.NORTH),
    PushES(ActionType.Push, MoveDirection.EAST, MoveDirection.SOUTH),

    PushWW(ActionType.Push, MoveDirection.WEST, MoveDirection.WEST),
    PushWN(ActionType.Push, MoveDirection.WEST, MoveDirection.NORTH),
    PushWS(ActionType.Push, MoveDirection.WEST, MoveDirection.SOUTH),


    //Pull
    PullNS(ActionType.Pull, MoveDirection.NORTH, MoveDirection.SOUTH),
    PullNE(ActionType.Pull, MoveDirection.NORTH, MoveDirection.EAST),
    PullNW(ActionType.Pull, MoveDirection.NORTH, MoveDirection.WEST),


    PullSN(ActionType.Pull, MoveDirection.SOUTH, MoveDirection.NORTH),
    PullSE(ActionType.Pull, MoveDirection.SOUTH, MoveDirection.EAST),
    PullSW(ActionType.Pull, MoveDirection.SOUTH, MoveDirection.WEST),

    PullEN(ActionType.Pull, MoveDirection.EAST, MoveDirection.NORTH),
    PullES(ActionType.Pull, MoveDirection.EAST, MoveDirection.SOUTH),
    PullEW(ActionType.Pull, MoveDirection.EAST, MoveDirection.WEST),

    PullWN(ActionType.Pull, MoveDirection.WEST, MoveDirection.NORTH),
    PullWS(ActionType.Pull, MoveDirection.WEST, MoveDirection.SOUTH),
    PullWE(ActionType.Pull, MoveDirection.WEST, MoveDirection.EAST),


    PaintN(ActionType.Paint,MoveDirection.NONE, MoveDirection.NORTH),
    PaintS(ActionType.Paint,MoveDirection.NONE, MoveDirection.SOUTH),
    PaintE(ActionType.Paint,MoveDirection.NONE, MoveDirection.EAST),
    PaintW(ActionType.Paint,MoveDirection.NONE, MoveDirection.WEST);


    private String name;
    private final ActionType type;
    private final MoveDirection agentMoveDirection;
    private final MoveDirection boxMoveDirection;

    Action(ActionType type, MoveDirection agentMoveDirection, MoveDirection boxMoveDirection) {
        this.type = type;
        this.agentMoveDirection = agentMoveDirection;
        this.boxMoveDirection = boxMoveDirection;
        //Setting name for enum
        name = generateName();
    }

    private String generateName(){
        String temp = type.name();
        if(agentMoveDirection != MoveDirection.NONE){
            temp += "(" + agentMoveDirection.getLetter();
            if(boxMoveDirection != MoveDirection.NONE){
                temp += ", " + boxMoveDirection.getLetter();
            }
            temp += ")";
        }
        return temp;
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
        for(Action a: Action.values()){
            if(a.getName().equals(action)) return a;
        }
        return NoOp;
    }


}

@Getter
enum MoveDirection {
    NORTH("N",-1, 0),
    EAST("E",0, 1),
    SOUTH("S",1, 0),
    WEST("W", 0, -1),
    NONE("", 0,0);

     private String letter;
     private final short deltaRow;
     private final short deltaCol;

    MoveDirection(String letter, int deltaRow, int deltaCol) {
        this.letter = letter;
        this.deltaRow = (short) deltaRow;
        this.deltaCol = (short) deltaCol;
    }

    public static MoveDirection fromShortString(String s) {
        for (MoveDirection moveDirection : MoveDirection.values()) {
            if (moveDirection.getLetter().equals(s)) return moveDirection;
        }
        return NONE;
    }
}

