package domain.gridworld.hospital2.state.objects.ui;

import domain.gridworld.hospital2.state.objects.Goal;
import lombok.Getter;

import java.awt.*;

import static domain.gridworld.hospital2.state.Colors.*;

public class GUIGoal extends GUIObject {
    @Getter private short row, col;

    public GUIGoal(String id, Character letter, Color color, short row, short col) {
        super(id, letter, color);
        this.row = row;
        this.col = col;
    }

    public GUIGoal(Goal goal) {
        super(goal.getId(), goal.getLetter(), goal.getColor());
        this.row = goal.getRow();
        this.col = goal.getCol();
    }

    @Override
    public void draw(Graphics2D g, CanvasDetails canvasDetails, short col, short row) {
        this.draw(g, canvasDetails, false);
    }

    void draw(Graphics2D g, CanvasDetails canvasDetails, boolean solved) {
        int top = canvasDetails.getOriginTop() + row * canvasDetails.getCellSize();
        int left = canvasDetails.getOriginLeft() + col * canvasDetails.getCellSize();
        int size = canvasDetails.getCellSize() - 2;
        g.setColor(solved ? GOAL_SOLVED_COLOR : GOAL_COLOR);
        if (this.isAgent()) {
            g.fillOval(left + 1, top + 1, size, size);
        } else {
            g.fillRect(left + 1, top + 1, size, size);
        }

        if (!solved) {
            g.setColor(GOAL_FONT_COLOR);
            this.letterText.draw(g, left + this.letterLeftOffset, top + this.letterTopOffset);
        }
    }
}
