package domain.gridworld.hospital2.state.objects;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class Coordinate implements Cloneable {
    private short row, col;

    public Coordinate(int row, int col) {
        this.row = (short) row;
        this.col = (short) col;
    }

    @Override
    public java.lang.Object clone() {
        return new Coordinate(this.row, this.col);
    }
}
