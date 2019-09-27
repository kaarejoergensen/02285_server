package domain.gridworld.hospital2.state.objects;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

import java.awt.*;

@Getter
@Setter
@AllArgsConstructor
public abstract class Object implements Cloneable {
    protected String id;
    protected char letter;
    protected short row, col;
    protected Color color;

    @Override
    public abstract java.lang.Object clone();

    public boolean sameCoordinates(Object otherObject) {
        return this.row == otherObject.row && this.col == otherObject.col;
    }
}
