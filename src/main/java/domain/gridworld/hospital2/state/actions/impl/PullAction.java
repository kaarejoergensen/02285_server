package domain.gridworld.hospital2.state.actions.impl;

import domain.gridworld.hospital2.state.State;
import domain.gridworld.hospital2.state.objects.Agent;
import domain.gridworld.hospital2.state.objects.Coordinate;
import domain.gridworld.hospital2.state.objects.StaticState;
import shared.Action;

public class PullAction extends MoveBoxAction {
    public PullAction(Action action, Agent agent, State state) {
        super(action, agent, state);
    }

    @Override
    Coordinate getBoxCoordinate() {
        return new Coordinate(
                agent.getCoordinate().getRow() + action.getBoxDeltaRow(),
                agent.getCoordinate().getCol() + action.getBoxDeltaCol());
    }

    @Override
    Coordinate getNewBoxCoordinate() {
        return new Coordinate(agent.getCoordinate().getRow(), agent.getCoordinate().getCol());
    }

    @Override
    public boolean isPreconditionsMet(State state, StaticState staticState) {
        return super.isPreconditionsMet(state, staticState) &&
                this.box != null && this.box.getColor().equals(this.agent.getColor()) &&
                staticState.getMap().isCell(this.newBoxCoordinate);
    }
}
