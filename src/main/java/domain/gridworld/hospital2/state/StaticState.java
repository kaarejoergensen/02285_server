package domain.gridworld.hospital2.state;

import domain.gridworld.hospital2.Object;
import lombok.AllArgsConstructor;
import lombok.Data;

import java.util.List;

@Data
public class StaticState {
    private String levelName;
    private String clientName;

    short numRows;
    short numCols;

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
}
