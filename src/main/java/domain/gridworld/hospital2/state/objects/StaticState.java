package domain.gridworld.hospital2.state.objects;

import domain.gridworld.hospital2.state.State;
import domain.gridworld.hospital2.state.objects.stateobjects.Goal;
import domain.gridworld.hospital2.state.objects.stateobjects.Object;
import lombok.Data;

import java.awt.*;
import java.util.List;
import java.util.Optional;

@Data
public class StaticState {
    private String levelName;
    private String clientName;

    private short numRows;
    private short numCols;
    private byte numAgents;

    private Map map;
    private List<Goal> agentGoals;
    private List<Goal> boxGoals;
    private List<Goal> allGoals;
    private int[] nextBoxColorMappingTable;

    public boolean isSolved(State state) {
        return this.agentGoals.stream().allMatch(a -> this.isSolved(state, a))
                && this.agentGoals.stream().allMatch(b -> this.isSolved(state, b));
    }

    private boolean isSolved(State state, Goal goal) {
        return this.isSolved(state, goal.getCoordinate(), goal.getLetter());
    }

    private boolean isSolved(State state, Coordinate coordinate, char letter) {
        Optional<? extends Object> object;
        if (isAgentGoal(letter)) {
            object = state.getAgentAt(coordinate);
        } else {
            object = state.getBoxAt(coordinate);
        }
        return object.isPresent() && object.get().getLetter() == letter;
    }

    private boolean isAgentGoal(Character letter) {
        return '0' <= letter && letter <= '9';
    }

    public void drawMap(Graphics2D g, CanvasDetails canvasDetails, int bufferWidth, int bufferHeight) {
        this.map.draw(g, canvasDetails, bufferWidth, bufferHeight);
    }

    public void initGoals(CanvasDetails canvasDetails) {
        this.allGoals.forEach(goal -> goal.letterTextUpdate(canvasDetails));
    }

    public void drawAllGoals(Graphics2D g, CanvasDetails canvasDetails) {
        this.allGoals.forEach(goal -> goal.draw(g, canvasDetails, false));
    }

    public void drawSolvedGoals(Graphics2D g, CanvasDetails canvasDetails, State currentState) {
        this.allGoals.stream().filter(goal -> this.isSolved(currentState, goal))
                .forEach(goal -> goal.draw(g, canvasDetails, true));
    }

    public void drawGoalsUnsolvedInNextState(Graphics2D g, CanvasDetails canvasDetails, State currentState, State nextState) {
        this.allGoals.stream().
                filter(goal -> this.isSolved(currentState, goal) && !this.isSolved(nextState, goal))
                .forEach(goal -> goal.draw(g, canvasDetails, false));
    }
}
