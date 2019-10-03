package domain.gridworld.hospital2;

import domain.gridworld.hospital2.state.State;
import domain.gridworld.hospital2.state.objects.StaticState;
import domain.gridworld.hospital2.state.objects.ui.CanvasDetails;
import lombok.Data;

import java.awt.*;

@Data
class GUIHandler {
    private CanvasDetails canvasDetails;
    private StaticState staticState;

    GUIHandler(StaticState staticState) {
        this.canvasDetails = new CanvasDetails();
        this.staticState = staticState;
    }

    void drawBackground(Graphics2D g, int bufferWidth, int bufferHeight, State state) {
        this.calculateCanvas(g, bufferWidth, this.staticState.getNumCols(), bufferHeight, this.staticState.getNumRows(), state);
        this.staticState.drawMap(g, this.canvasDetails, bufferWidth, bufferHeight);
        this.staticState.drawAllGoals(g, this.canvasDetails);
    }

    private void calculateCanvas(Graphics2D g, int bufferWidth, int numCols, int bufferHeight, int numRows, State state) {
        this.canvasDetails.calculate(g, bufferWidth, numCols, bufferHeight, numRows);
        state.initObjects(this.canvasDetails);
        this.staticState.initGoals(this.canvasDetails);
    }

    void drawStateBackground(Graphics2D g, State currentState) {
        this.staticState.drawSolvedGoals(g, this.canvasDetails, currentState);
        if (currentState.getStateTime() != 0.0) currentState.drawStaticObjects(g, this.canvasDetails);
    }

    void drawStateTransition(Graphics2D g, State currentState, State nextState, double interpolation) {
        this.staticState.drawGoalsUnsolvedInNextState(g, this.canvasDetails, currentState, nextState);
        if (currentState.getStateTime() != 0.0) {
            currentState.drawDynamicObjects(g, this.canvasDetails, nextState, interpolation);
        } else {
            currentState.drawAllObjects(g, this.canvasDetails, nextState, interpolation);
        }
    }
}
