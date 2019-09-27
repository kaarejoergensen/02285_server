package domain.gridworld.hospital2.state.objects;

import domain.gridworld.hospital2.state.State;
import domain.gridworld.hospital2.state.objects.ui.GUIGoal;
import lombok.Data;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@Data
public class StaticState {
    private String levelName;
    private String clientName;

    short numRows;
    short numCols;
    byte numAgents;

    private Map map;
    private List<Goal> agentGoals;
    private List<Goal> boxGoals;

    public boolean isSolved(State state) {
        return this.agentGoals.stream().allMatch(a -> this.isSolved(state, a))
                && this.agentGoals.stream().allMatch(b -> this.isSolved(state, b));
    }

    public boolean isSolved(State state, GUIGoal goal) {
        return this.isSolved(state, goal.getCol(), goal.getRow(), goal.getLetter());
    }

    private boolean isSolved(State state, Goal goal) {
        return this.isSolved(state, goal.getCol(), goal.getRow(), goal.getLetter());
    }

    private boolean isSolved(State state, short col, short row, char letter) {
        Optional<? extends Object> object;
        if (isAgentGoal(letter)) {
            object = state.getAgentAt(col, row);
        } else {
            object = state.getBoxAt(col, row);
        }
        return object.isPresent() && object.get().getLetter() == letter;
    }

    public List<Goal> getAllGoals() {
        return Stream.concat(this.agentGoals.stream(), this.boxGoals.stream()).collect(Collectors.toList());
    }

    private boolean isAgentGoal(Character letter) {
        return '0' <= letter && letter <= '9';
    }
}
