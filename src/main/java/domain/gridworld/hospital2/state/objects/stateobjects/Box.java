package domain.gridworld.hospital2.state.objects.stateobjects;

import domain.gridworld.hospital2.state.objects.Coordinate;
import domain.gridworld.hospital2.state.objects.CanvasDetails;
import lombok.*;

import java.awt.*;

@Data
public class Box extends Object {
    private NextColor nextColor;

    public Box(String id, char letter, Coordinate coordinate, NextColor nextColor) {
        super(id, letter, coordinate, nextColor.color);
        this.nextColor = nextColor;
    }

    private Box(String id, char letter, Coordinate coordinate, NextColor nextColor, LetterTextContainer letterText) {
        super(id, letter, coordinate, nextColor.color, letterText);
        this.nextColor = nextColor;
    }

    @Override
    public void draw(Graphics2D g, CanvasDetails canvasDetails, Coordinate newCoordinate, double interpolation, Color color) {
        Color newColor = this.blendColors(this.color, color, interpolation);
        super.draw(g, canvasDetails, newCoordinate, interpolation, newColor);
    }

    @Override
    public java.lang.Object clone() {
        return new Box(id, letter, (Coordinate) coordinate.clone(), nextColor, letterText);
    }

    @Getter
    @Setter
    @AllArgsConstructor
    @NoArgsConstructor
    public static class NextColor {
        private Color color;
        private NextColor next;

        @Override
        public String toString() {
            return "Color: " + color + " Next color: " + next.getColor();
        }
    }
}
