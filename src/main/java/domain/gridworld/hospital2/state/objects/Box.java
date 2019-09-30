package domain.gridworld.hospital2.state.objects;

import java.awt.*;

public class Box extends Object {
    public Box(String id, char letter, short row, short col, Color color) {
        super(id, letter, row, col, color);
    }

    private Box(String id, char letter, short row, short col, Color color, LetterTextContainer letterText) {
        super(id, letter, row, col, color, letterText);
    }

    @Override
    public java.lang.Object clone() {
        return new Box(id, letter, row, col, color, letterText);
    }
}
