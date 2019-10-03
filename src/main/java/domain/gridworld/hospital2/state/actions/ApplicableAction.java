package domain.gridworld.hospital2.state.actions;

import domain.gridworld.hospital2.state.objects.Agent;
import domain.gridworld.hospital2.state.objects.Coordinate;
import lombok.AllArgsConstructor;
import shared.Action;

@AllArgsConstructor
public abstract class ApplicableAction implements IApplicableAction {
    protected Action action;
    protected Agent agent;

    @Override
    public boolean isConflicting(IApplicableAction other) {
        for (Coordinate coordinate : this.getPostCoordinates()) {
            for (Coordinate otherCoordinate : other.getPostCoordinates()) {
                if (coordinate.equals(otherCoordinate)) return true;
            }
        }
        return false;
    }
}
