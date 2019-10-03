package domain.gridworld.hospital2.state.objects;

import domain.gridworld.hospital2.state.objects.ui.CanvasDetails;
import org.javatuples.Pair;
import shared.Farge;

import java.awt.*;


public class Goal extends Object {
    public Goal(String id, char letter, Coordinate coordinate, Color color) {
        super(id, letter, coordinate, color);
    }

    private Goal(String id, char letter, Coordinate coordinate, Color color, LetterTextContainer letterText) {
        super(id, letter, coordinate, color, letterText);
    }

    @Override
    public java.lang.Object clone() {
        return new Goal(id, letter, (Coordinate) coordinate.clone(), color, letterText);
    }

    @Override
    public void draw(Graphics2D g, CanvasDetails canvasDetails) {
        this.draw(g, canvasDetails, false);
    }

    public void draw(Graphics2D g, CanvasDetails canvasDetails, boolean solved) {
        Pair<Integer, Integer> coordinates = this.calculateCoordinates(canvasDetails, this.coordinate);
        int size = canvasDetails.getCellSize() - 2;
        g.setColor(solved ? Farge.GoalSolvedColor.color : Farge.GoalColor.color);
        if (this.isAgent()) {
            g.fillOval(coordinates.getValue1() + 1, coordinates.getValue0() + 1, size, size);
        } else {
            g.fillRect(coordinates.getValue1() + 1, coordinates.getValue0() + 1, size, size);
        }

        if (!solved) {
            g.setColor(Farge.GoalFontColor.color);
            this.letterText.draw(g, coordinates.getValue1(), coordinates.getValue0());
        }
    }
}
