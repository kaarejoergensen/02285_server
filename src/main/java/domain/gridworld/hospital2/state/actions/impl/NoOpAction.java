package domain.gridworld.hospital2.state.actions.impl;

import domain.gridworld.hospital2.state.State;
import domain.gridworld.hospital2.state.actions.ApplicableAction;
import domain.gridworld.hospital2.state.objects.stateobjects.Agent;
import domain.gridworld.hospital2.state.objects.Coordinate;
import domain.gridworld.hospital2.state.objects.stateobjects.Object;
import domain.gridworld.hospital2.state.objects.StaticState;
import domain.gridworld.hospital2.state.objects.CanvasDetails;

import java.awt.*;
import java.util.Collections;
import java.util.List;

public class NoOpAction extends ApplicableAction {

    public NoOpAction(Agent agent) {
        super(null, agent);
    }

    @Override
    public boolean isPreconditionsMet(State state, StaticState staticState) {
        return true;
    }

    @Override
    public List<Coordinate> getPostCoordinates() {
        return Collections.emptyList();
    }

    @Override
    public void apply(State newState) {
    }

    @Override
    public void draw(Graphics2D g, CanvasDetails canvasDetails, State nextState, double interpolation) {
    }

    @Override
    public List<Object> getAffectedObjects() {
        return Collections.emptyList();
    }
}
