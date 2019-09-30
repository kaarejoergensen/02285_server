package domain.gridworld.hospital2.state.objects.ui;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.javatuples.Pair;

import java.awt.*;
import java.awt.font.TextLayout;

import static domain.gridworld.hospital2.state.Colors.BOX_AGENT_FONT_COLOR;

@RequiredArgsConstructor
@Getter
public abstract class GUIObject {
    final protected String id;

    final protected Character letter;
    final protected Color color;
    protected TextLayout letterText;
    protected int letterTopOffset, letterLeftOffset;

    void letterTextUpdate(CanvasDetails canvasDetails){
        this.letterText = new TextLayout(String.valueOf(this.letter), canvasDetails.getCurrentFont(), canvasDetails.getFontRenderContext());
        Rectangle bound = this.letterText.getPixelBounds(canvasDetails.getFontRenderContext(), 0, 0);

        int size = canvasDetails.getCellSize() - 2 * canvasDetails.getCellTextMargin();
        this.letterTopOffset = canvasDetails.getCellTextMargin() + size - (size - bound.height) / 2;
        this.letterLeftOffset = canvasDetails.getCellTextMargin() + (size - bound.width) / 2 - bound.x;
    }

    public void draw(Graphics2D g, CanvasDetails canvasDetails, short row, short col) {
        var coordinates = this.calculateCoordinates(canvasDetails, row, col);
        this.draw(g, canvasDetails, coordinates);
    }

    public void draw(Graphics2D g, CanvasDetails canvasDetails, short row, short col, short newRow, short newCol, double interpolation) {
        var coordinates = this.calculateInterpolationCoordinates(canvasDetails, row, col, newRow, newCol, interpolation);
        this.draw(g, canvasDetails, coordinates);
    }

    private void draw(Graphics2D g, CanvasDetails canvasDetails, Pair<Integer, Integer> coordinates) {
        int size = canvasDetails.getCellSize() - 2 * canvasDetails.getCellBoxMargin();
        g.setColor(this.color);
        if (this.isAgent()) {
            g.fillOval(coordinates.getValue1() + canvasDetails.getCellBoxMargin(), coordinates.getValue0() + canvasDetails.getCellBoxMargin(), size, size);
        } else {
            g.fillRect(coordinates.getValue1() + canvasDetails.getCellBoxMargin(), coordinates.getValue0() + canvasDetails.getCellBoxMargin(), size, size);
        }

        g.setColor(BOX_AGENT_FONT_COLOR);
        letterText.draw(g, coordinates.getValue1() + this.letterLeftOffset, coordinates.getValue0() + this.letterTopOffset);
    }

    Pair<Integer, Integer> calculateInterpolationCoordinates(CanvasDetails canvasDetails,
                                                             short row, short col,
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
}
