package domain.gridworld.hospital2.state.objects;

import domain.gridworld.hospital2.state.objects.ui.CanvasDetails;
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
public abstract class Object implements Cloneable {
    protected String id;
    protected char letter;
    protected short row, col;
    protected Color color;

    protected LetterTextContainer letterText;

    Object(String id, char letter, short row, short col, Color color) {
        this.id = id;
        this.letter = letter;
        this.row = row;
        this.col = col;
        this.color = color;
        this.letterText = new LetterTextContainer();
    }

    public boolean sameCoordinates(Object other) {
        return this.row == other.row && this.col == other.col;
    }

    @Override
    public abstract java.lang.Object clone();

    public void letterTextUpdate(CanvasDetails canvasDetails){
        this.letterText.letterTextUpdate(canvasDetails, this.letter);
    }

    public void draw(Graphics2D g, CanvasDetails canvasDetails) {
        var coordinates = this.calculateCoordinates(canvasDetails, this.row, this.col);
        this.draw(g, canvasDetails, coordinates, this.color);
    }

    public void draw(Graphics2D g, CanvasDetails canvasDetails,  short newRow, short newCol, double interpolation) {
        var coordinates = this.calculateInterpolationCoordinates(canvasDetails, this.row, this.col, newRow, newCol, interpolation);
        this.draw(g, canvasDetails, coordinates, this.color);
    }

    public void draw(Graphics2D g, CanvasDetails canvasDetails,  short newRow, short newCol, double interpolation, Color color) {
        var coordinates = this.calculateInterpolationCoordinates(canvasDetails, this.row, this.col, newRow, newCol, interpolation);
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

    Pair<Integer, Integer> calculateInterpolationCoordinates(CanvasDetails canvasDetails, short row, short col,
                                                             short newRow, short newCol, double interpolation) {
        Pair<Integer, Integer> oldCoordinates = this.calculateCoordinates(canvasDetails, row, col);
        Pair<Integer, Integer> newCoordinates = this.calculateCoordinates(canvasDetails, newRow, newCol);

        int interpolationTop = this.calculateInterpolation(oldCoordinates.getValue0(), newCoordinates.getValue0(), interpolation);
        int interpolationLeft = this.calculateInterpolation(oldCoordinates.getValue1(), newCoordinates.getValue1(), interpolation);

        return Pair.with(interpolationTop, interpolationLeft);
    }

    Pair<Integer, Integer> calculateCoordinates(CanvasDetails canvasDetails, short row, short col) {
        return Pair.with(this.calculateTop(canvasDetails, row), this.calculateLeft(canvasDetails, col));
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
