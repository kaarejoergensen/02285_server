package domain.gridworld.hospital2.state.objects.ui;

import domain.gridworld.hospital2.state.objects.Box;

import java.awt.*;

public class GUIBox extends GUIObject {
    public GUIBox(String id, Character letter, Color color) {
        super(id, letter, color);
    }

    public GUIBox(Box box) {
        super(box.getId(), box.getLetter(), box.getColor());
    }
}
