package domain.gridworld.hospital2.state.objects.stateobjects;

import domain.gridworld.hospital2.state.objects.Coordinate;
import domain.gridworld.hospital2.state.objects.CanvasDetails;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.javatuples.Pair;
import shared.Farge;

import java.awt.*;
import java.awt.font.TextLayout;


@Getter
@Setter
@AllArgsConstructor
public abstract class Object implements Cloneable, Comparable {
    protected String id;
    protected char letter;
    protected Coordinate coordinate;
    protected Color color;

    protected LetterTextContainer letterText;

    Object(String id, char letter, Coordinate coordinate, Color color) {
        this.id = id;
        this.letter = letter;
        this.coordinate = coordinate;
        this.color = color;
        this.letterText = new LetterTextContainer();
    }

    @Override
    public abstract java.lang.Object clone();

    public void letterTextUpdate(CanvasDetails canvasDetails){
        this.letterText.letterTextUpdate(canvasDetails, this.letter);
    }

    public void draw(Graphics2D g, CanvasDetails canvasDetails) {
        var coordinates = this.calculateCoordinates(canvasDetails, this.coordinate);
        this.draw(g, canvasDetails, coordinates, this.color);
    }

    public void draw(Graphics2D g, CanvasDetails canvasDetails,  Coordinate newCoordinate, double interpolation) {
        var coordinates = this.calculateInterpolationCoordinates(canvasDetails, this.coordinate, newCoordinate, interpolation);
        this.draw(g, canvasDetails, coordinates, this.color);
    }

    public void draw(Graphics2D g, CanvasDetails canvasDetails,  Coordinate newCoordinate, double interpolation, Color color) {
        var coordinates = this.calculateInterpolationCoordinates(canvasDetails, this.coordinate, newCoordinate, interpolation);
        this.draw(g, canvasDetails, coordinates, color);
    }

    private void draw(Graphics2D g, CanvasDetails canvasDetails, Pair<Integer, Integer> coordinates, Color color) {
        int size = canvasDetails.getCellSize() - 2 * canvasDetails.getCellBoxMargin();
        g.setColor(color);
        if (this.isAgent()) {
            g.fillOval(coordinates.getValue1() + canvasDetails.getCellBoxMargin(), coordinates.getValue0() + canvasDetails.getCellBoxMargin(), size, size);
        } else {
            g.fillRect(coordinates.getValue1() + canvasDetails.getCellBoxMargin(), coordinates.getValue0() + canvasDetails.getCellBoxMargin(), size, size);
        }

        g.setColor(Farge.BoxAgentFontColor.color);
        letterText.draw(g, coordinates.getValue1(), coordinates.getValue0());
    }

    Pair<Integer, Integer> calculateInterpolationCoordinates(CanvasDetails canvasDetails, Coordinate coordinate,
                                                             Coordinate newCoordinate, double interpolation) {
        Pair<Integer, Integer> oldCoordinates = this.calculateCoordinates(canvasDetails, coordinate);
        Pair<Integer, Integer> newCoordinates = this.calculateCoordinates(canvasDetails, newCoordinate);

        int interpolationTop = this.calculateInterpolation(oldCoordinates.getValue0(), newCoordinates.getValue0(), interpolation);
        int interpolationLeft = this.calculateInterpolation(oldCoordinates.getValue1(), newCoordinates.getValue1(), interpolation);

        return Pair.with(interpolationTop, interpolationLeft);
    }

    Pair<Integer, Integer> calculateCoordinates(CanvasDetails canvasDetails, Coordinate coordinate) {
        return Pair.with(this.calculateTop(canvasDetails, coordinate.getRow()), this.calculateLeft(canvasDetails, coordinate.getCol()));
    }

    private int calculateTop(CanvasDetails canvasDetails, short row) {
        return canvasDetails.getOriginTop() + row * canvasDetails.getCellSize();
    }

    private int calculateLeft(CanvasDetails canvasDetails, short col) {
        return canvasDetails.getOriginLeft() + col * canvasDetails.getCellSize();
    }

    int calculateInterpolation(int oldN, int newN, double interpolation) {
        return (int) (oldN + (newN - oldN) * interpolation);
    }

    boolean isAgent() {
        return '0' <= this.letter && this.letter <= '9';
    }

    Color blendColors(Color c1, Color c2, double ratio) {
        int r = (int) ((1.0 - ratio) * c1.getRed() + ratio * c2.getRed());
        int g = (int) ((1.0 - ratio) * c1.getGreen() + ratio * c2.getGreen());
        int b = (int) ((1.0 - ratio) * c1.getBlue() + ratio * c2.getBlue());
        return new Color(r, g, b);
    }

    @Override
    public int compareTo(java.lang.Object o) {
        return this.id.compareTo(((Object) o).getId());
    }

    @Getter
    @Setter
    @NoArgsConstructor
    public static class LetterTextContainer {
        protected TextLayout letterText;
        protected int letterTopOffset, letterLeftOffset;

        void letterTextUpdate(CanvasDetails canvasDetails, char letter){
            this.letterText = new TextLayout(String.valueOf(letter), canvasDetails.getCurrentFont(), canvasDetails.getFontRenderContext());
            Rectangle bound = this.letterText.getPixelBounds(canvasDetails.getFontRenderContext(), 0, 0);

            int size = canvasDetails.getCellSize() - 2 * canvasDetails.getCellTextMargin();
            this.letterTopOffset = canvasDetails.getCellTextMargin() + size - (size - bound.height) / 2;
            this.letterLeftOffset = canvasDetails.getCellTextMargin() + (size - bound.width) / 2 - bound.x;
        }

        public void draw(Graphics2D g, int left, int top) {
            this.letterText.draw(g, left + this.letterLeftOffset, top + this.letterTopOffset);
        }
    }
}
