package searchclient.level;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class Coordinate {
    private short row, col;

    public Coordinate(int row, int col) {
        this.row = (short) row;
        this.col = (short) col;
    }
}
