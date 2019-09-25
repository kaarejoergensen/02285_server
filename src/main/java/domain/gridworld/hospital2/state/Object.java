package domain.gridworld.hospital2.state;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.awt.*;

@Data
@AllArgsConstructor
public class Object implements Cloneable {
    private int id;
    private char letter;
    private short row;
    private short col;
    private Color color;

    @Override
    protected java.lang.Object clone() {
        return new Object(id, letter, row, col, color);
    }
}
