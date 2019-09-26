package domain.gridworld.hospital2.state.objects;

import java.awt.*;

public class Box extends Object {
    public Box(int id, char letter, short row, short col, Color color) {
        super(id, letter, row, col, color);
    }

    @Override
    public java.lang.Object clone() {
        return new Box(id, letter, row, col, color);
    }
}
