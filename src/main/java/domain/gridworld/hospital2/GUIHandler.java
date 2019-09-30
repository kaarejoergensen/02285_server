package domain.gridworld.hospital2;

import domain.gridworld.hospital2.state.State;
import domain.gridworld.hospital2.state.objects.StaticState;
import domain.gridworld.hospital2.state.objects.ui.CanvasDetails;
import lombok.Data;

import java.awt.*;

@Data
public class GUIHandler {
    private CanvasDetails canvasDetails;

    private StaticState staticState;

    private boolean staticObjectsDrawn;

    public GUIHandler(StaticState staticState) {
        this.canvasDetails = new CanvasDetails();
        this.staticState = staticState;
    }

    public void drawBackground(Graphics2D g, int bufferWidth, int numCols, int bufferHeight, int numRows, State state) {
        this.calculateCanvas(g, bufferWidth, numCols, bufferHeight, numRows, state);
        this.staticState.drawMap(g, canvasDetails, bufferWidth, bufferHeight);
        this.staticState.drawAllGoals(g, canvasDetails);
    }

    //TODO: Fix bug in first state, where static objects are not shown
    public void drawStateBackground(Graphics2D g, StaticState staticState, State currentState, State nextState) {
        staticState.drawSolvedGoals(g, canvasDetails, currentState);
        if (nextState != null) nextState.drawStaticObjects(g, this.canvasDetails);
        this.staticObjectsDrawn = nextState != null;
    }

    public void drawStateTransition(Graphics2D g, StaticState staticState,
                                    State currentState, State nextState, double interpolation) {
        staticState.drawGoalsUnsolvedInNextState(g, canvasDetails, currentState, nextState);
        if (!this.staticObjectsDrawn) {
            currentState.drawStaticObjects(g, this.canvasDetails);
            this.staticObjectsDrawn = true;
        }
        currentState.drawDynamicObjects(g, this.canvasDetails, nextState, interpolation);
    }

    private void calculateCanvas(Graphics2D g, int bufferWidth, int numCols, int bufferHeight, int numRows, State state) {
        this.canvasDetails.calculate(g, bufferWidth, numCols, bufferHeight, numRows);
        state.initObjects(this.canvasDetails);
        this.staticState.initGoals(this.canvasDetails);
    }
}
