package domain.gridworld.hospital2.state.objects.ui;

import domain.gridworld.hospital2.state.State;
import domain.gridworld.hospital2.state.objects.Object;
import domain.gridworld.hospital2.state.objects.*;
import lombok.Data;

import java.awt.*;
import java.util.Collection;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@Data
public class GUIState {
    private CanvasDetails canvasDetails;

    private Map map;

    private java.util.Map<String, GUIAgent> agents;
    private java.util.Map<String, GUIBox> boxes;
    private List<GUIGoal> goals;

    public GUIState(Map map, Collection<Agent> agents, Collection<Box> boxes, Collection<Goal> goals) {
        this.canvasDetails = new CanvasDetails();
        this.map = map;
        this.agents = agents.stream().map(GUIAgent::new).collect(Collectors.toMap(GUIAgent::getId, Function.identity()));
        this.boxes = boxes.stream().map(GUIBox::new).collect(Collectors.toMap(GUIBox::getId, Function.identity()));
        this.goals = goals.stream().map(GUIGoal::new).collect(Collectors.toList());
    }

    private void calculateCanvas(Graphics2D g, int bufferWidth, int numCols, int bufferHeight, int numRows) {
        this.canvasDetails.calculate(g, bufferWidth, numCols, bufferHeight, numRows);
        this.boxes.values().forEach(b -> b.letterTextUpdate(this.canvasDetails));
        this.agents.values().forEach(a -> {
            a.letterTextUpdate(this.canvasDetails);
            a.setArmsSize(this.canvasDetails);
        });
        this.goals.forEach(goal -> goal.letterTextUpdate(this.canvasDetails));
    }

    public void drawBackground(Graphics2D g, int bufferWidth, int numCols, int bufferHeight, int numRows) {
        this.calculateCanvas(g, bufferWidth, numCols, bufferHeight, numRows);
        this.map.draw(g, canvasDetails, bufferWidth, bufferHeight);
        this.goals.forEach(goal -> goal.draw(g, this.canvasDetails, false));
    }

    public void drawStateBackground(Graphics2D g, StaticState staticState, State currentState, State nextState) {
        this.drawSolvedGoals(g, staticState, currentState);
        if (nextState != null) this.drawStaticObjects(g, currentState, nextState);
    }

    private void drawSolvedGoals(Graphics2D g, StaticState staticState, State currentState) {
        this.goals.stream().filter(goal -> staticState.isSolved(currentState, goal))
                .forEach(goal -> goal.draw(g, this.canvasDetails, true));
    }

    private void drawStaticObjects(Graphics2D g, State currentState, State nextState) {
        for (Object object : this.getStaticObjects(currentState, nextState)) {
            if (object instanceof Agent) {
                this.agents.get(object.getId()).draw(g, this.canvasDetails, object.getRow(), object.getCol());
            } else if (object instanceof Box) {
                this.boxes.get(object.getId()).draw(g, this.canvasDetails, object.getRow(), object.getCol());
            }
        }
    }

    private List<? extends Object> getStaticObjects(State currentState, State nextState) {
        Stream<? extends Object> staticAgents = currentState.getAgents().stream().
                filter(a -> a.sameCoordinates(nextState.getAgent(a.getId())));
        Stream<? extends Object> staticBoxes = currentState.getBoxes().stream().
                filter(b -> b.sameCoordinates(nextState.getBox(b.getId())));
        return Stream.concat(staticAgents, staticBoxes).collect(Collectors.toList());
    }
}
