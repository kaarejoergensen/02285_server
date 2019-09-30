package domain.gridworld.hospital2.state.objects.ui;

import domain.gridworld.hospital2.state.objects.Goal;
import lombok.Getter;
import org.javatuples.Pair;

import java.awt.*;

import static domain.gridworld.hospital2.state.Colors.*;

public class GUIGoal extends GUIObject {
    @Getter private short row, col;

    GUIGoal(Goal goal) {
        super(goal.getId(), goal.getLetter(), goal.getColor());
        this.row = goal.getRow();
        this.col = goal.getCol();
    }

    @Override
    public void draw(Graphics2D g, CanvasDetails canvasDetails, short col, short row) {
        this.draw(g, canvasDetails, false);
    }

    void draw(Graphics2D g, CanvasDetails canvasDetails, boolean solved) {
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
