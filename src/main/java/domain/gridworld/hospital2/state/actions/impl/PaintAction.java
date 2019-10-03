package domain.gridworld.hospital2.state.actions.impl;

import domain.gridworld.hospital2.state.State;
import domain.gridworld.hospital2.state.actions.ApplicableAction;
import domain.gridworld.hospital2.state.objects.Agent;
import domain.gridworld.hospital2.state.objects.Box;
import domain.gridworld.hospital2.state.objects.Coordinate;
import domain.gridworld.hospital2.state.objects.Object;
import domain.gridworld.hospital2.state.objects.StaticState;
import domain.gridworld.hospital2.state.objects.ui.CanvasDetails;
import shared.Action;
import shared.Farge;

import java.awt.*;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class PaintAction extends ApplicableAction {
    private Box box;
    private Color newColor;

    public PaintAction(Action action, Agent agent, State state) {
        super(action, agent);
        Coordinate boxCoordinate = new Coordinate(
                agent.getCoordinate().getRow() + action.getBoxDeltaRow(),
                agent.getCoordinate().getCol() + action.getBoxDeltaCol());
        this.box = state.getBoxAt(boxCoordinate).orElse(null);
        if (this.box != null) this.newColor = Farge.next(this.box.getColor()).color;
    }

    @Override
    public boolean isPreconditionsMet(State state, StaticState staticState) {
        return this.box != null && this.agent.getColor().equals(Farge.Grey.color);
    }

    @Override
    public List<Coordinate> getPostCoordinates() {
        return Collections.emptyList();
    }

    @Override
    public void apply(State newState) {
        Optional<Box> newBox = newState.getBoxAt(this.box.getCoordinate());
        newBox.ifPresent(value -> value.setColor(this.newColor));
    }

    @Override
    public void draw(Graphics2D g, CanvasDetails canvasDetails, State nextState, double interpolation) {
        this.agent.drawArmPaint(g, canvasDetails, this.box.getCoordinate(), this.box.getColor(), this.newColor, interpolation);
        this.agent.draw(g, canvasDetails, this.agent.getCoordinate(), interpolation);
        this.box.draw(g, canvasDetails, this.box.getCoordinate(), interpolation, newColor);
    }

    @Override
    public List<Object> getAffectedObjects() {
        return Stream.of(this.agent, this.box).collect(Collectors.toList());
    }
}
