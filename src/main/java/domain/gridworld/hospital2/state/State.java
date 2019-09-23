package domain.gridworld.hospital2.state;

import domain.gridworld.hospital2.Action;
import domain.gridworld.hospital2.Object;
import lombok.*;

import java.util.ArrayList;
import java.util.List;

@Data
@AllArgsConstructor
public class State {
    private List<Object> boxes;
    private List<Object> agents;

    private long stateTime;

    public Object getBox(int id) {
        return this.boxes.get(id);
    }

    public Object getAgent(int id) {
        return this.agents.get(id);
    }

    public State copyOf() {
        return new State(new ArrayList<>(this.boxes), new ArrayList<>(this.agents), -1);
    }

    public void applyAction(int agent, Action action) {

    }
}
