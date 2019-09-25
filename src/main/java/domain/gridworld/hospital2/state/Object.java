package domain.gridworld.hospital2.state;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.awt.*;

@Data
@AllArgsConstructor
public class Object {
    private int id;
    private char letter;
    private short row;
    private short col;
    private Color color;
}
