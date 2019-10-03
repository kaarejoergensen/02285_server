package domain.gridworld.hospital2.state.actions.impl;

import domain.gridworld.hospital2.state.State;
import domain.gridworld.hospital2.state.actions.ApplicableAction;
import domain.gridworld.hospital2.state.objects.Agent;
import domain.gridworld.hospital2.state.objects.Coordinate;
import domain.gridworld.hospital2.state.objects.Object;
import domain.gridworld.hospital2.state.objects.StaticState;
import domain.gridworld.hospital2.state.objects.ui.CanvasDetails;
import shared.Action;

import java.awt.*;
import java.util.Collections;
import java.util.List;

public class MoveAction extends ApplicableAction {
    Coordinate newAgentCoordinate;

    public MoveAction(Action action, Agent agent) {
        super(action, agent);
        short newAgentRow = (short) (agent.getCoordinate().getRow() + action.getAgentDeltaRow());
        short newAgentCol = (short) (agent.getCoordinate().getCol() + action.getAgentDeltaCol());
        this.newAgentCoordinate = new Coordinate(newAgentRow, newAgentCol);
    }

    @Override
    public boolean isPreconditionsMet(State state, StaticState staticState) {
        return state.isCellFree(this.newAgentCoordinate) && staticState.getMap().isCell(this.newAgentCoordinate);
    }

    @Override
    public List<Coordinate> getPostCoordinates() {
        return Collections.singletonList(this.newAgentCoordinate);
    }

    @Override
    public void apply(State newState) {
        newState.getAgent(this.agent.getId()).setCoordinate(this.newAgentCoordinate);
    }

    @Override
    public void draw(Graphics2D g, CanvasDetails canvasDetails, State oldState, State nextState, double interpolation) {
        Agent nextAgent = nextState.getAgent(this.agent.getId());
        this.agent.drawArmMove(g, canvasDetails, nextAgent.getCoordinate(), interpolation);
        this.agent.draw(g, canvasDetails, nextAgent.getCoordinate(), interpolation);
    }

    @Override
    public List<Object> getAffectedObjects() {
        return Collections.singletonList(this.agent);
    }

}
