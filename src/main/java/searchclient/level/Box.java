package searchclient.level;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import shared.Farge;

import java.util.Objects;

@RequiredArgsConstructor
@Getter
public class Box {
    private int _hash = 0;

    private final char character;
    private final Farge color;

    public boolean equals(final Object o) {
        if (o == this) return true;
        if (!(o instanceof Box)) return false;
        final Box other = (Box) o;
        return this.hashCode() == other.hashCode() && this.character == other.character && Objects.equals(this.color, other.color);
    }

    public int hashCode() {
        if (this._hash == 0) {
            final int PRIME = 59;
            int result = 1;
            result = result * PRIME + this.character;
            final Object $color = this.getColor();
            result = result * PRIME + ($color == null ? 43 : $color.hashCode());
            this._hash = result;
        }
        return this._hash;
    }
}
