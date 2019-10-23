package domain.gridworld.hospital2.state.actions.impl;

import domain.gridworld.hospital2.state.State;
import domain.gridworld.hospital2.state.objects.stateobjects.Agent;
import domain.gridworld.hospital2.state.objects.Coordinate;
import domain.gridworld.hospital2.state.objects.StaticState;
import shared.Action;

public class PushAction extends MoveBoxAction {
    public PushAction(Action action, Agent agent, State state) {
        super(action, agent, state);
    }

    @Override
    Coordinate getBoxCoordinate() {
        return new Coordinate(
                agent.getCoordinate().getRow() + action.getAgentDeltaRow(),
                agent.getCoordinate().getCol() + action.getAgentDeltaCol());
    }

    @Override
    Coordinate getNewBoxCoordinate() {
        return new Coordinate(
                box.getCoordinate().getRow() + action.getBoxDeltaRow(),
                box.getCoordinate().getCol() + action.getBoxDeltaCol());
    }

    @Override
    public boolean isPreconditionsMet(State state, StaticState staticState) {
        return staticState.getMap().isCell(this.newAgentCoordinate) &&
                this.box != null && this.box.getColor().equals(this.agent.getColor()) &&
                staticState.getMap().isCell(this.newBoxCoordinate) &&
                state.isCellFree(this.newBoxCoordinate);
    }
}
