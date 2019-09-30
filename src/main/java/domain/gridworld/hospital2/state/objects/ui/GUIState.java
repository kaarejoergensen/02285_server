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

    private boolean staticObjectsDrawn;

    public GUIState(Map map, Collection<Agent> agents, Collection<Box> boxes, Collection<Goal> goals) {
        this.canvasDetails = new CanvasDetails();
        this.map = map;
        this.agents = agents.stream().map(GUIAgent::new).collect(Collectors.toMap(GUIAgent::getId, Function.identity()));
        this.boxes = boxes.stream().map(GUIBox::new).collect(Collectors.toMap(GUIBox::getId, Function.identity()));
        this.goals = goals.stream().map(GUIGoal::new).collect(Collectors.toList());
    }

    public void drawBackground(Graphics2D g, int bufferWidth, int numCols, int bufferHeight, int numRows) {
        this.calculateCanvas(g, bufferWidth, numCols, bufferHeight, numRows);
        this.map.draw(g, canvasDetails, bufferWidth, bufferHeight);
        this.goals.forEach(goal -> goal.draw(g, this.canvasDetails, false));
    }

    public void drawStateBackground(Graphics2D g, StaticState staticState, State currentState, State nextState) {
        this.drawSolvedGoals(g, staticState, currentState);
        if (nextState != null) this.drawStaticObjects(g, nextState);
        this.staticObjectsDrawn = nextState != null;
    }

    public void drawStateTransition(Graphics2D g, StaticState staticState,
                                    State currentState, State nextState, double interpolation) {
        this.drawGoalsUnsolvedInNextState(g, staticState, currentState, nextState);
        if (!this.staticObjectsDrawn) {
            this.drawStaticObjects(g, currentState);
            this.staticObjectsDrawn = true;
        }
        this.drawDynamicObjects(g, currentState, nextState, interpolation);
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

    private void drawSolvedGoals(Graphics2D g, StaticState staticState, State currentState) {
        this.goals.stream().filter(goal -> staticState.isSolved(currentState, goal))
                .forEach(goal -> goal.draw(g, this.canvasDetails, true));
    }

    private void drawGoalsUnsolvedInNextState(Graphics2D g, StaticState staticState, State currentState, State nextState) {
        this.goals.stream().
                filter(goal -> staticState.isSolved(currentState, goal) && !staticState.isSolved(nextState, goal))
                .forEach(goal -> goal.draw(g, this.canvasDetails, false));
    }

    private void drawStaticObjects(Graphics2D g, State nextState) {
        for (Object object : this.getStaticObjects(nextState)) {
            if (object instanceof Agent) {
                this.agents.get(object.getId()).draw(g, this.canvasDetails, object.getRow(), object.getCol());
            } else if (object instanceof Box) {
                this.boxes.get(object.getId()).draw(g, this.canvasDetails, object.getRow(), object.getCol());
            }
        }
    }

    private List<? extends Object> getStaticObjects(State state) {
        Stream<? extends Object> staticAgents = state.getAgents().stream().
                filter(a -> !state.getMovedAgents().contains(a.getId()));
        Stream<? extends Object> staticBoxes = state.getBoxes().stream().
                filter(b -> !state.getMovedBoxes().contains(b.getId()));
        return Stream.concat(staticAgents, staticBoxes).collect(Collectors.toList());
    }

    private void drawDynamicObjects(Graphics2D g, State currentState, State nextState, double interpolation) {
        for (String agentId : nextState.getMovedAgents()) {
            Agent oldAgent = currentState.getAgent(agentId);
            Agent newAgent = nextState.getAgent(agentId);
            GUIAgent guiAgent = this.agents.get(agentId);
            if (interpolation != 0.0) this.drawAgentArm(g, currentState, nextState, interpolation, newAgent, oldAgent, guiAgent);
            guiAgent.draw(g, this.canvasDetails, oldAgent.getRow(), oldAgent.getCol(),
                    newAgent.getRow(), newAgent.getCol(), interpolation);
        }
        for (String boxId : nextState.getMovedBoxes()) {
            Box oldBox = currentState.getBox(boxId);
            Box newBox = nextState.getBox(boxId);
            this.boxes.get(boxId).draw(g, this.canvasDetails, oldBox.getRow(), oldBox.getCol(),
                    newBox.getRow(), newBox.getCol(), interpolation);
        }
    }

    private void drawAgentArm(Graphics2D g, State currentState, State nextState, double interpolation,
                              Agent newAgent, Agent oldAgent, GUIAgent guiAgent) {
        for (String boxId : nextState.getMovedBoxes()) {
            Box oldBox = currentState.getBox(boxId);
            Box newBox = nextState.getBox(boxId);
            if (newAgent.getRow() == oldBox.getRow() && newAgent.getCol() == oldBox.getCol() ||
                    oldAgent.getRow() == newBox.getRow() && oldAgent.getCol() == newBox.getCol()) {
                guiAgent.drawArmPullPush(g, this.canvasDetails, oldAgent, newAgent, oldBox, newBox, interpolation);
                return;
            }
        }
        guiAgent.drawArmMove(g, this.canvasDetails, oldAgent, newAgent, interpolation);
    }
}
