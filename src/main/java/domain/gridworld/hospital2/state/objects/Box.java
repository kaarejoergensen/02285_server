package domain.gridworld.hospital2.state.objects;

import domain.gridworld.hospital2.state.objects.ui.CanvasDetails;

import java.awt.*;

public class Box extends Object {
    public Box(String id, char letter, Coordinate coordinate, Color color) {
        super(id, letter, coordinate, color);
    }

    private Box(String id, char letter, Coordinate coordinate, Color color, LetterTextContainer letterText) {
        super(id, letter, coordinate, color, letterText);
    }

    @Override
    public void draw(Graphics2D g, CanvasDetails canvasDetails, Coordinate newCoordinate, double interpolation, Color color) {
        Color newColor = this.blendColors(this.color, color, interpolation);
        super.draw(g, canvasDetails, newCoordinate, interpolation, newColor);
    }

    @Override
    public java.lang.Object clone() {
        return new Box(id, letter, (Coordinate) coordinate.clone(), color, letterText);
    }
}
