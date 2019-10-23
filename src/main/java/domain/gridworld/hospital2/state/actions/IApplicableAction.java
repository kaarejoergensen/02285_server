package domain.gridworld.hospital2.state.actions;

import domain.gridworld.hospital2.state.State;
import domain.gridworld.hospital2.state.actions.impl.*;
import domain.gridworld.hospital2.state.objects.stateobjects.Agent;
import domain.gridworld.hospital2.state.objects.Coordinate;
import domain.gridworld.hospital2.state.objects.stateobjects.Object;
import domain.gridworld.hospital2.state.objects.StaticState;
import domain.gridworld.hospital2.state.objects.CanvasDetails;
import shared.Action;

import java.awt.*;
import java.util.List;

public interface IApplicableAction {

    static IApplicableAction getInstance(Action action, Agent agent, State state) {
        switch (action.getType()) {
            case Move:
                return new MoveAction(action, agent);
            case Push:
                return new PushAction(action, agent, state);
            case Pull:
                return new PullAction(action, agent, state);
            case Paint:
                return new PaintAction(action, agent, state);
            default:
                return new NoOpAction(agent);
        }
    }

    boolean isPreconditionsMet(State state, StaticState staticState);

    boolean isConflicting(IApplicableAction other);

    List<Coordinate> getPostCoordinates();

    void apply(State newState);

    void draw(Graphics2D g, CanvasDetails canvasDetails, State nextState, double interpolation);

    List<Object> getAffectedObjects();
}
