package domain.gridworld.hospital2.state.actions.impl;

import domain.gridworld.hospital2.state.State;
import domain.gridworld.hospital2.state.objects.stateobjects.Agent;
import domain.gridworld.hospital2.state.objects.stateobjects.Box;
import domain.gridworld.hospital2.state.objects.Coordinate;
import domain.gridworld.hospital2.state.objects.stateobjects.Object;
import domain.gridworld.hospital2.state.objects.CanvasDetails;
import shared.Action;

import java.awt.*;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public abstract class MoveBoxAction extends MoveAction {
    protected Box box;
    Coordinate newBoxCoordinate;

    MoveBoxAction(Action action, Agent agent, State state) {
        super(action, agent);
        Coordinate boxCoordinate = this.getBoxCoordinate();
        this.box = state.getBoxAt(boxCoordinate).orElse(null);
        if (this.box != null) {
            this.newBoxCoordinate = this.getNewBoxCoordinate();
        }
    }

    abstract Coordinate getBoxCoordinate();

    abstract Coordinate getNewBoxCoordinate();

    @Override
    public List<Coordinate> getPostCoordinates() {
        return Stream.of(this.newAgentCoordinate, this.newBoxCoordinate).collect(Collectors.toList());
    }

    @Override
    public void apply(State newState) {
        super.apply(newState);
        Optional<Box> newBox = newState.getBoxAt(this.box.getCoordinate());
        newBox.ifPresent(value -> newState.updateBox(value, this.newBoxCoordinate));
    }

    @Override
    public void draw(Graphics2D g, CanvasDetails canvasDetails, State nextState, double interpolation) {
        this.agent.drawArmPullPush(g, canvasDetails, this.newAgentCoordinate, this.box.getCoordinate(),
                this.newBoxCoordinate, interpolation);
        this.agent.draw(g, canvasDetails, this.newAgentCoordinate, interpolation);
        this.box.draw(g, canvasDetails, this.newBoxCoordinate, interpolation);
    }

    @Override
    public List<Object> getAffectedObjects() {
        return Stream.of(this.agent, this.box).collect(Collectors.toList());
    }
}
