package searchclient.level;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class Coordinate {
    private int _hash = 0;
    private short row, col;

    public Coordinate(int row, int col) {
        this.row = (short) row;
        this.col = (short) col;
    }

    public boolean equals(final Object o) {
        if (o == this) return true;
        if (!(o instanceof Coordinate)) return false;
        final Coordinate other = (Coordinate) o;
        return this.hashCode() == other.hashCode();
    }

    public int hashCode() {
        if (this._hash == 0) {
            long bits = 7L;
            bits = 31L * bits + row;
            bits = 31L * bits + col;
            this._hash = (int) (bits ^ (bits >> 32));
        }
        return this._hash;
    }
}
