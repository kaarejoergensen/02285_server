package domain.gridworld.hospital2.state.objects;

import java.awt.*;

public class Goal extends Object {
    public Goal(String id, char letter, short row, short col, Color color) {
        super(id, letter, row, col, color);
    }

    @Override
    public java.lang.Object clone() {
        return new Goal(id, letter, row, col, color);
    }


}
