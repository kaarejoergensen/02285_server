package domain.gridworld.hospital2.state.objects;

import java.awt.*;

public class Agent extends Object {

    public Agent(String id, char letter, short row, short col, Color color) {
        super(id, letter, row, col, color);
    }

    @Override
    public java.lang.Object clone() {
        return new Agent(id, letter, row, col, color);
    }
}
