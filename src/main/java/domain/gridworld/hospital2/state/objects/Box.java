package domain.gridworld.hospital2.state.objects;

import domain.gridworld.hospital2.state.objects.ui.CanvasDetails;

import java.awt.*;

public class Box extends Object {
    public Box(String id, char letter, short row, short col, Color color) {
        super(id, letter, row, col, color);
    }

    private Box(String id, char letter, short row, short col, Color color, LetterTextContainer letterText) {
        super(id, letter, row, col, color, letterText);
    }

    @Override
    public void draw(Graphics2D g, CanvasDetails canvasDetails, short newRow, short newCol, double interpolation, Color color) {
        Color newColor = this.blendColors(this.color, color, interpolation);
        super.draw(g, canvasDetails, newRow, newCol, interpolation, newColor);
    }

    @Override
    public java.lang.Object clone() {
        return new Box(id, letter, row, col, color, letterText);
    }

    private Color blendColors(Color c1, Color c2, double ratio)
    {
        int r = (int) ((1.0 - ratio) * c1.getRed() + ratio * c2.getRed());
        int g = (int) ((1.0 - ratio) * c1.getGreen() + ratio * c2.getGreen());
        int b = (int) ((1.0 - ratio) * c1.getBlue() + ratio * c2.getBlue());
        return new Color(r, g, b);
    }
}
