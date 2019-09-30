package domain.gridworld.hospital2.state.objects;

import domain.gridworld.hospital2.state.objects.ui.CanvasDetails;
import org.javatuples.Pair;

import java.awt.*;
import java.awt.font.TextLayout;

import static domain.gridworld.hospital2.state.Colors.*;

public class Goal extends Object {
    public Goal(String id, char letter, short row, short col, Color color) {
        super(id, letter, row, col, color);
    }

    private Goal(String id, char letter, short row, short col, Color color, TextLayout letterText, int letterTopOffset, int letterLeftOffset) {
        super(id, letter, row, col, color, letterText, letterTopOffset, letterLeftOffset);
    }

    @Override
    public java.lang.Object clone() {
        return new Goal(id, letter, row, col, color, letterText, letterTopOffset, letterLeftOffset);
    }

    @Override
    public void draw(Graphics2D g, CanvasDetails canvasDetails) {
        this.draw(g, canvasDetails, false);
    }

    public void draw(Graphics2D g, CanvasDetails canvasDetails, boolean solved) {
        Pair<Integer, Integer> coordinates = this.calculateCoordinates(canvasDetails, this.row, this.col);
        int size = canvasDetails.getCellSize() - 2;
        g.setColor(solved ? GOAL_SOLVED_COLOR : GOAL_COLOR);
        if (this.isAgent()) {
            g.fillOval(coordinates.getValue1() + 1, coordinates.getValue0() + 1, size, size);
        } else {
            g.fillRect(coordinates.getValue1() + 1, coordinates.getValue0() + 1, size, size);
        }

        if (!solved) {
            g.setColor(GOAL_FONT_COLOR);
            this.letterText.draw(g, coordinates.getValue1() + this.letterLeftOffset, coordinates.getValue0() + this.letterTopOffset);
        }
    }
}
