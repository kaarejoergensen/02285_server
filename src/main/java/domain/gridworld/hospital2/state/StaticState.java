package domain.gridworld.hospital2.state;

import lombok.Data;
import searchclient.NotImplementedException;

import java.util.List;
import java.util.Optional;

@Data
public class StaticState {
    private String levelName;
    private String clientName;

    short numRows;
    short numCols;
    byte numAgents;

    private List<List<Boolean>> map;
    private List<Object> agentGoals;
    private List<Object> boxGoals;

    public boolean isCell(int row, int col) {
        return this.isPartOfMap(row, col) && map.get(row).get(col);
    }

    public boolean isWall(int row, int col) {
        return this.isPartOfMap(row, col) && !map.get(row).get(col);
    }

    private boolean isPartOfMap(int row, int col) {
        return row < map.size() && col < map.get(row).size();
    }

    public boolean isSolved(State state) {
        for (Object goal : agentGoals) {
            Optional<Object> agent = state.getAgentAt(goal.getCol(), goal.getRow());
            if (agent.isEmpty() || agent.get().getLetter() != goal.getLetter()) return false;
        }
        for (Object goal : boxGoals) {
            Optional<Object> box = state.getBoxAt(goal.getCol(), goal.getRow());
            if (box.isEmpty() || box.get().getLetter() != goal.getLetter()) return false;
        }
        return true;
    }
}
